"""Param-schema fetch + dynamic ``FillTemplate`` model, with a short TTL cache.

Template fill needs a Pydantic model whose fields are *this template's* Mustache
parameters (1:1), so the model can be forced as a tool and its output validated.
That model can't be static — it differs per template — so it is built at query
time from the template's **param-schema**, a metadata doc ml-commons derives at
registration (from the Mustache body's parse tree + the index mapping) and stores
one-per-template in the ``.plugins-ml-agentic-search-templates`` system index.

This module owns two things:
  1. :func:`build_fill_model` — turn a param-schema dict into a typed Pydantic
     model (enums → ``Literal``, required → no default, optional → ``| None``).
  2. :class:`TemplateSchemaCache` — read the schema doc for a ``template_id`` and
     cache the built model in memory with a short TTL, so a schema edit takes at
     most one TTL to take effect and a hot template is a one-entry lookup. Never
     persisted.

The schema is read per request the way mappings are read today; the agent server
stays a stateless control plane that owns no template state of its own.
"""

from __future__ import annotations

import keyword
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ConfigDict, Field
from pydantic import create_model as build_pydantic_model

from agents.agentic_search.prompts.template_fill import FILL_MODEL_NAME

logger = logging.getLogger(__name__)

# ml-commons system index holding one param-schema doc per template, keyed by
# ``template_id`` (also the core ``_scripts`` template name). Must match the ml-commons
# CommonValue.ML_AGENTIC_SEARCH_TEMPLATES_INDEX constant.
AGENTIC_SEARCH_TEMPLATES_INDEX = ".plugins-ml-agentic-search-templates"

# How long a built model is trusted before the next query re-reads the schema doc.
DEFAULT_SCHEMA_TTL_SECONDS = 60.0

# Synthetic abstain field on every ``FillTemplate`` model (not a real Mustache param):
# when the question needs a capability the template can't express, the model sets this
# instead of force-filling params into valid-but-wrong DSL, and the strategy routes to
# the free-DSL fallback. Optional, so it costs ~0 output tokens on the happy path.
#
# The enumerated triggers below are load-bearing: vaguer ("set true when unsure") and
# heavier ("reason step by step first") wordings both over-abstained and lost accuracy
# in a prompt sweep. Re-run the sweep before editing.
CANNOT_EXPRESS_FIELD = "cannot_express"
_CANNOT_EXPRESS_DESCRIPTION = (
    "Set true when the question needs a capability NOT among these parameters. "
    "Concretely set it true if the question: (a) restricts text matching to ONE "
    "specific field (e.g. 'in the title', 'in the name') and no parameter isolates that "
    "field; (b) demands an EXACT contiguous phrase / literal wording in a field and no "
    "phrase parameter exists; (c) asks to RANK or BOOST by a signal (most popular, "
    "trending, boost recent/newer, custom relevance) and no parameter or sort option "
    "expresses that ranking; (d) asks for a COUNT-only answer, aggregation, faceting, or "
    "grouping; (e) references a field, similarity ('products like X'), or predicate that "
    "has no matching parameter. Otherwise leave it false and fill the parameters."
)

# param-schema "type" -> Python annotation for the non-enum case. ``number`` is
# ``int | float``, int first, so ``size:5`` renders as ``5``; a bare ``float`` would
# coerce it to ``5.0``, which OpenSearch rejects for integer fields.
_SCALAR_TYPES: dict[str, Any] = {
    "string": str,
    "text": str,
    "keyword": str,
    "integer": int,
    "int": int,
    "long": int,
    "number": int | float,
    "float": float,
    "double": float,
    "boolean": bool,
    "bool": bool,
}


@dataclass(frozen=True)
class TemplateSchema:
    """A template's param-schema and the Pydantic model built from it."""

    template_id: str
    index_binding: str | None
    param_schema: dict[str, Any]
    fill_model: type


def _annotation_for(name: str, spec: dict[str, Any]) -> Any:
    """Return the type annotation for one param, from its schema entry.

    An ``enum`` becomes a ``Literal`` (an illegal value is then impossible); a bare
    ``type`` maps to a Python scalar. Unknown types fall back to ``str`` — the model
    can still emit a value and rendering stays the arbiter.
    """
    if "enum" in spec:
        enum = spec["enum"]
        if not isinstance(enum, list) or not enum:
            raise ValueError(f"param '{name}' has an empty or non-list enum")
        # Literal accepts a tuple of literal values (subscription form).
        return Literal[tuple(enum)]
    type_name = str(spec.get("type", "string")).lower()
    py_type = _SCALAR_TYPES.get(type_name)
    if py_type is None:
        logger.warning(
            "param '%s' has unknown type '%s'; treating as string", name, type_name
        )
        py_type = str
    return py_type


def _safe_field_name(param_name: str, used: set[str]) -> str:
    """Return a valid, unique Python identifier for a param name.

    A param name comes from a Mustache placeholder and can be anything the template
    author wrote — dotted (``author.first_name``), leading-underscore, a Python
    keyword, or a ``model_``/``model_config`` name that collides with Pydantic's
    reserved namespace. Any of those would crash or shadow ``create_model``. We map
    each to a safe field name and carry the real name as the field's ``alias`` so the
    model still emits (and validates) the original key.
    """
    safe = re.sub(r"\W", "_", param_name)
    # Field names may not start with a digit or underscore (underscore fields are
    # treated as private by Pydantic and rejected by create_model).
    if not safe or not (safe[0].isalpha()):
        safe = "f_" + safe
    if keyword.iskeyword(safe):
        safe = safe + "_"
    # Ensure uniqueness after collapsing (e.g. "a.b" and "a-b" both -> "a_b").
    candidate = safe
    n = 1
    while candidate in used:
        candidate = f"{safe}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def build_fill_model(
    param_schema: dict[str, Any], *, model_name: str = FILL_MODEL_NAME
) -> type:
    """Build a Pydantic model whose fields are the template's params, 1:1.

    Each param entry is ``{type, required?, enum?, description?}``. Required params
    are non-nullable with no default (the model must supply them); optional params
    are ``T | None`` defaulting to ``None`` so an omitted param simply disappears in
    the Mustache body's optional sections.

    Field names are sanitized to valid Python identifiers with the real param name
    kept as the field ``alias`` (so an arbitrary Mustache name can't crash model
    construction); callers must dump ``by_alias=True`` to recover the real names.
    ``protected_namespaces=()`` disables Pydantic's ``model_*`` guard so a param like
    ``model_id`` registers normally. ``extra="ignore"`` (not ``forbid``) so a single
    hallucinated key doesn't fail the whole fill — an unknown key is simply dropped,
    and the render-parse guard still backstops a bad result.

    Raises:
        ValueError: The schema is empty or a param entry is malformed.
    """
    if not param_schema:
        raise ValueError("param_schema is empty; nothing to fill")

    fields: dict[str, tuple[Any, Any]] = {}
    used_names: set[str] = set()
    # Reserve the abstain field's name so no real param can be sanitized onto it.
    # A template whose author literally named a param ``cannot_express`` keeps that
    # param (it maps to a distinct safe name) and simply forgoes the escape hatch.
    add_abstain = CANNOT_EXPRESS_FIELD not in param_schema
    if add_abstain:
        used_names.add(CANNOT_EXPRESS_FIELD)
    else:
        logger.warning(
            "template has a real param named '%s'; abstain escape hatch disabled",
            CANNOT_EXPRESS_FIELD,
        )
    for name, spec in param_schema.items():
        if not isinstance(spec, dict):
            raise ValueError(f"param '{name}' schema entry must be an object")
        annotation = _annotation_for(name, spec)
        description = spec.get("description", "")
        field_name = _safe_field_name(name, used_names)
        if spec.get("required", False):
            fields[field_name] = (
                annotation,
                Field(description=description, alias=name),
            )
        else:
            # Optional: nullable + default None -> omitted from the tool's
            # `required` and safely absent when the model doesn't fill it.
            fields[field_name] = (
                annotation | None,
                Field(default=None, description=description, alias=name),
            )

    # Synthetic abstain field: optional bool, default False. Not a Mustache param, so
    # the strategy strips it before rendering. Named identically to its alias (a plain
    # identifier), so it survives ``model_dump(by_alias=True)`` for the strategy to read.
    if add_abstain:
        fields[CANNOT_EXPRESS_FIELD] = (
            bool,
            Field(default=False, description=_CANNOT_EXPRESS_DESCRIPTION),
        )

    model = build_pydantic_model(
        model_name,
        __config__=ConfigDict(
            extra="ignore",
            populate_by_name=True,
            protected_namespaces=(),
        ),
        **fields,
    )
    return model


class TemplateSchemaCache:
    """Reads param-schema docs and caches the built models with a short TTL.

    Keyed by ``template_id``. The cached value is the built model + schema, not any
    client or credential, so it is safe to share across requests; the TTL bounds how
    long a schema edit takes to take effect. A single-process, last-writer-wins dict
    — adequate for a stateless control plane (each worker keeps its own).
    """

    def __init__(self, *, ttl_seconds: float = DEFAULT_SCHEMA_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[TemplateSchema, float]] = {}

    def get(self, template_id: str, client: Any) -> TemplateSchema:
        """Return the ``TemplateSchema`` for ``template_id``, fetching if stale.

        Uses ``client`` (the per-request, caller-authenticated OpenSearch client) to
        read the schema doc when the cache misses or the entry has expired.

        Raises:
            ValueError: No schema doc for ``template_id`` (unregistered template),
                or the doc has no usable ``param_schema``.
        """
        now = time.monotonic()
        cached = self._cache.get(template_id)
        if cached is not None and now < cached[1]:
            return cached[0]

        schema = self._fetch_and_build(template_id, client)
        self._cache[template_id] = (schema, now + self._ttl)
        return schema

    def _fetch_and_build(self, template_id: str, client: Any) -> TemplateSchema:
        doc = self._fetch_doc(template_id, client)
        if doc is None:
            raise ValueError(
                f"no param-schema registered for template_id '{template_id}'"
            )
        param_schema = doc.get("param_schema")
        if not isinstance(param_schema, dict) or not param_schema:
            raise ValueError(
                f"param-schema for template_id '{template_id}' is missing or empty"
            )
        fill_model = build_fill_model(param_schema)
        logger.info(
            "Built FillTemplate model for template_id=%s (%d params)",
            template_id,
            len(param_schema),
        )
        return TemplateSchema(
            template_id=template_id,
            index_binding=doc.get("index_binding"),
            param_schema=param_schema,
            fill_model=fill_model,
        )

    @staticmethod
    def _fetch_doc(template_id: str, client: Any) -> dict[str, Any] | None:
        """Fetch the schema doc from the system index. Returns None if absent.

        The doc ``_id`` is the ``template_id``. Uses the typed ``get`` call; a
        missing doc (404) surfaces as an exception from opensearch-py, which we
        translate to ``None`` so the caller raises a clear "unregistered" error.
        """
        try:
            resp = client.get(index=AGENTIC_SEARCH_TEMPLATES_INDEX, id=template_id)
        except Exception as e:  # noqa: BLE001 - 404 / index-missing -> unregistered
            logger.warning(
                "Schema-doc fetch for template_id=%s failed: %s", template_id, e
            )
            return None
        if not resp or not resp.get("found"):
            return None
        return resp.get("_source") or None
