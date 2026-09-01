"""Multi-template fill: choose one of several candidate templates and fill it.

Selected when a request carries more than one candidate in ``context.template_ids``.
One forced tool call both picks a template and fills its parameters: every candidate's
params are merged into one flat object, namespaced by template id and all optional,
alongside a required ``template_id`` choice. The chosen template's params are then
validated against its real model (which restores the required-ness and enum constraints
an all-optional merge cannot express) and rendered by OpenSearch, exactly as in the
single-template path.

The combined call is used only when there is a choice to make:

- one candidate (after filtering) -> the single-template strategy, whose prompt is
  dedicated to filling one template;
- several candidates -> the combined pick-and-fill call.

Candidates are pre-filtered to the request's index and capped, because the merged schema
grows with the candidate count and an unbounded set would dominate the prompt.

Any failure, or an abstention, degrades to the free-DSL fallback rather than returning a
wrong query, the same contract as :mod:`template_fill`.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from agents.agentic_search.prompts.multi_template_fill import (
    MULTI_FILL_MODEL_NAME,
    MULTI_FILL_SYSTEM_BLOCKS,
    MULTI_FILL_USER_PROMPT,
    NONE_CHOICE,
    format_candidates,
)
from agents.agentic_search.strategies.base import GenerationRequest, GenerationStrategy
from agents.agentic_search.strategies.forced_tool import forced_tool_fill_raw
from agents.agentic_search.strategies.template_base import (
    TEMPLATE_ID_KEY,
    TEMPLATE_IDS_KEY,
    TemplateStrategyBase,
    distinct_template_ids,
)
from agents.agentic_search.strategies.template_fill import TemplateFillStrategy
from agents.agentic_search.template_schema import (
    CANNOT_EXPRESS_FIELD,
    TemplateSchema,
    TemplateSchemaCache,
    build_fill_model,
    json_type_for,
)

logger = logging.getLogger(__name__)


class _NoCandidateExpresses(Exception):
    """The model declined: no candidate template can express the question.

    Raised on the happy path so :meth:`MultiTemplateFillStrategy.generate` routes to
    the free-DSL fallback -- the same handling as a structural failure, but triggered by
    the model's own judgment rather than a broken render.
    """


# The merged tool schema's choice property. Value-identical to TEMPLATE_ID_KEY but a
# distinct concept: this names a field in the LLM-facing tool schema, not a request
# context key, so the two must stay independent.
CHOICE_FIELD = "template_id"

# Upper bound on candidates fed to one call. The merged schema grows with the candidate
# count, so an unbounded set would crowd out the question and blow past the point where
# prompt caching keeps it affordable. Extra candidates are dropped with a warning rather
# than silently, so a caller can see the set was truncated.
MAX_CANDIDATES = 8


def _is_true(value: Any) -> bool:
    """Whether a raw tool-call value means boolean true.

    The forced tool's input is decoded JSON, not a validated model, so a flag may arrive
    as a real boolean or as a string. Plain truthiness would read ``"false"`` as true and
    silently abstain on every request, so accept only genuine true spellings.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return value is not None and value is not False and value == 1


def _param_json_schema(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON Schema for one param entry of a stored param-schema."""
    out: dict[str, Any] = {}
    if spec.get("description"):
        out["description"] = spec["description"]
    raw_type = str(spec.get("type", "string")).lower()
    enum = spec.get("enum")
    if isinstance(enum, list) and enum:
        out["enum"] = list(enum)
        # Take the declared type rather than assuming string: the validating model turns
        # the same members into a Literal preserving their original types, so declaring
        # a numeric enum as a string would make the model emit "5" and then fail
        # validation against Literal[1, 5, 10].
        out["type"] = json_type_for(raw_type)
        if not all(isinstance(v, str) for v in enum) and out["type"] == "string":
            # Mixed or non-string members with no usable declared type: omit the type
            # and let the enum alone constrain the value.
            out.pop("type", None)
        return out
    out["type"] = json_type_for(raw_type)
    if raw_type == "array":
        hint = 'JSON array literal, e.g. ["a","b"].'
        out["description"] = f"{out.get('description', '')} {hint}".strip()
    return out


def _prefix_for(template_id: str) -> str:
    """Sanitized namespace prefix for one template id, before disambiguation.

    Params are namespaced because identically-named slots from different candidates
    would otherwise collide on a single merged property, letting the model fill the
    wrong template's value into it. Use :func:`_prefixes_for` for prefixes that are also
    unique across a candidate set.
    """
    return "".join(c if c.isalnum() else "_" for c in template_id) + "__"


def _prefixes_for(template_ids: list[str]) -> dict[str, str]:
    """Map each template id to a prefix that is unique within this candidate set.

    Sanitizing is lossy, so distinct ids can collapse onto the same prefix (``a-b`` and
    ``a_b`` both become ``a_b__``), which would let one template's parameter schema
    silently overwrite another's. A collision gets a positional suffix so every
    candidate keeps its own namespace.
    """
    out: dict[str, str] = {}
    used: set[str] = set()
    for position, template_id in enumerate(template_ids):
        prefix = _prefix_for(template_id)
        if prefix in used:
            prefix = f"{prefix[:-2]}_{position}__"
        # Pathological ids could still collide after suffixing; walk until unique.
        while prefix in used:
            prefix = f"{prefix[:-2]}_x__"
        used.add(prefix)
        out[template_id] = prefix
    return out


class MultiTemplateFillStrategy(TemplateStrategyBase):
    """Pick one of several candidate templates and fill it in a single call.

    The constructor, the render step, the free-DSL fallback and the inherited
    ``needs_mapping = False`` come from
    :class:`~agents.agentic_search.strategies.template_base.TemplateStrategyBase`.
    """

    name = "multi_template_fill"

    def __init__(
        self,
        *,
        single: GenerationStrategy | None = None,
        fallback: GenerationStrategy | None = None,
        schema_cache: TemplateSchemaCache | None = None,
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        # Resolve the shared defaults first: self._single is built from them below.
        super().__init__(fallback=fallback, schema_cache=schema_cache)
        # Single-template path, reused whenever only one candidate survives filtering.
        self._single = (
            single
            if single is not None
            else TemplateFillStrategy(
                fallback=self._fallback, schema_cache=self._schema_cache
            )
        )
        self._max_candidates = max_candidates

    # ---- entry point ------------------------------------------------------

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        ids = distinct_template_ids(request.context)
        if not ids:
            logger.warning(
                "multi_template_fill selected without candidates; falling back to free-DSL"
            )
            return self._fallback_generate(request)

        try:
            candidates = self._resolve(ids, request)
        except Exception as e:  # noqa: BLE001 - resolution failure degrades
            logger.warning("candidate resolution failed (%s); falling back", e)
            return self._fallback_generate(request)

        if not candidates:
            logger.warning(
                "no candidate template resolved for index=%s; falling back",
                request.index_name,
            )
            return self._fallback_generate(request)

        # One candidate needs no choice; use the dedicated single-template path.
        if len(candidates) == 1:
            return self._delegate_single(request, candidates[0].template_id)

        try:
            return self._select_and_fill(request, candidates)
        except _NoCandidateExpresses:
            logger.info(
                "no candidate can express the question (index=%s); routing to free-DSL",
                request.index_name,
            )
            return self._fallback_generate(request)
        except Exception as e:  # noqa: BLE001 - any failure degrades to free-DSL
            logger.warning(
                "multi-template fill failed for index=%s (%s); falling back to free-DSL",
                request.index_name,
                e,
            )
            return self._fallback_generate(request)

    # ---- candidate handling ----------------------------------------------

    def _resolve(
        self, ids: list[str], request: GenerationRequest
    ) -> list[TemplateSchema]:
        """Resolve candidates, dropping unusable ones and capping the set.

        A candidate that cannot be read (unregistered, or not readable by this caller)
        is skipped rather than failing the request: with several candidates, one bad id
        must not deny the others. Candidates bound to a different index are dropped
        because the rendered query runs against this request's index.
        """
        resolved: list[TemplateSchema] = []
        # Stop as soon as the cap is met. Each unresolved id costs a system-index read
        # and a model build, so capping only after the loop would let a caller's long
        # list drive that work regardless of how few candidates are actually used.
        for position, tid in enumerate(ids):
            if len(resolved) >= self._max_candidates:
                logger.warning(
                    "candidate set capped at %d; ignoring %s",
                    self._max_candidates,
                    ", ".join(ids[position:]),
                )
                break
            try:
                schema = self._schema_cache.get(tid, request.client)
            except Exception as e:  # noqa: BLE001 - skip this candidate only
                logger.warning("candidate %s unresolved (%s); skipping", tid, e)
                continue
            if (
                schema.index_binding
                and request.index_name
                and schema.index_binding != request.index_name
            ):
                logger.debug(
                    "candidate %s is bound to index %s, not %s; skipping",
                    tid,
                    schema.index_binding,
                    request.index_name,
                )
                continue
            resolved.append(schema)
        return resolved

    def _delegate_single(
        self, request: GenerationRequest, template_id: str
    ) -> dict[str, Any]:
        """Run the single-template strategy for ``template_id``."""
        context = {**request.context, TEMPLATE_ID_KEY: template_id}
        context.pop(TEMPLATE_IDS_KEY, None)
        return self._single.generate(replace(request, context=context))

    # ---- the combined call ------------------------------------------------

    def _select_and_fill(
        self, request: GenerationRequest, candidates: list[TemplateSchema]
    ) -> dict[str, Any]:
        """Choose a template and fill it in one forced tool call, then render."""
        tool_spec, name_map = self._build_tool_spec(candidates)
        filled = forced_tool_fill_raw(
            model=request.model,
            tool_spec=tool_spec,
            system_blocks=MULTI_FILL_SYSTEM_BLOCKS,
            user_message=MULTI_FILL_USER_PROMPT.format(
                question=request.question,
                candidates=format_candidates(
                    [(c.template_id, c.description or "") for c in candidates]
                ),
            ),
        )

        if _is_true(filled.pop(CANNOT_EXPRESS_FIELD, False)):
            raise _NoCandidateExpresses("no candidate can express the question")

        # Resolve the choice against the real candidates first, so a template whose id
        # happens to be the abstain sentinel is still selectable.
        choice = filled.get(CHOICE_FIELD)
        chosen = next((c for c in candidates if c.template_id == choice), None)
        if chosen is None:
            if not choice or choice == NONE_CHOICE:
                raise _NoCandidateExpresses("model chose no template")
            raise ValueError(f"model chose unknown template '{choice}'")

        # Select this template's values by exact key membership rather than by string
        # prefix: one candidate's prefix can be a prefix of another's (ids "x" and
        # "x__y"), which would otherwise pull in a sibling's parameters.
        keys = name_map[chosen.template_id]
        params = {
            keys[key]: value
            for key, value in filled.items()
            if key in keys and value is not None
        }
        # Validate against the template's real model: the merged schema marks every
        # param optional and cannot express enums per template, so required-ness and
        # allowed values are enforced here instead. Dropping unset params lets the
        # body's inverted sections and optional clauses behave as authored.
        model_cls = build_fill_model(chosen.param_schema, add_abstain=False)
        clean = model_cls.model_validate(params).model_dump(
            by_alias=True, exclude_none=True
        )

        rendered = self._render(request.client, chosen.template_id, clean)
        logger.info(
            "Multi-template fill chose %s of %d candidates and rendered %d params",
            chosen.template_id,
            len(candidates),
            len(clean),
        )
        return rendered

    def _build_tool_spec(
        self, candidates: list[TemplateSchema]
    ) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
        """Build the merged pick-and-fill tool schema.

        The abstain flag is declared before the choice so the model decides
        expressibility before committing to a template.

        Returns the tool spec plus, per template id, a map from emitted property name
        back to the real Mustache parameter name. Reading the fill through that map
        avoids inferring ownership from string prefixes, which is unsafe when one
        candidate's prefix is a prefix of another's.
        """
        ids = [c.template_id for c in candidates]
        props: dict[str, Any] = {
            CANNOT_EXPRESS_FIELD: {
                "type": "boolean",
                "description": (
                    "Set true when no candidate template can express the question — a "
                    "field, filter, projection, aggregation, count-only answer, exact "
                    "phrase, prefix/wildcard/fuzzy match, custom ranking, or similarity "
                    "that none of the parameters below cover. Leave false otherwise."
                ),
            },
            CHOICE_FIELD: {
                "type": "string",
                # Only add the sentinel when no candidate already uses that id, since a
                # duplicated enum member is invalid JSON Schema.
                "enum": ids + ([NONE_CHOICE] if NONE_CHOICE not in ids else []),
                "description": (
                    "Id of the template you are filling, or 'none'. Fill only the "
                    "parameters carrying this id's prefix."
                ),
            },
        }
        prefixes = _prefixes_for([c.template_id for c in candidates])
        name_map: dict[str, dict[str, str]] = {}
        for candidate in candidates:
            prefix = prefixes[candidate.template_id]
            keys: dict[str, str] = {}
            for name, spec in candidate.param_schema.items():
                if not isinstance(spec, dict):
                    continue
                key = f"{prefix}{name}"
                # A merged property must never be claimed by two templates.
                if key in props:
                    logger.warning(
                        "merged param name %s collides; skipping it for %s",
                        key,
                        candidate.template_id,
                    )
                    continue
                props[key] = _param_json_schema(spec)
                keys[key] = name
            name_map[candidate.template_id] = keys
        spec_out = {
            "name": MULTI_FILL_MODEL_NAME,
            "description": "Choose the best-fitting search template and fill its parameters.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": props,
                    "required": [CHOICE_FIELD],
                }
            },
        }
        return spec_out, name_map


__all__ = ["MAX_CANDIDATES", "MultiTemplateFillStrategy"]
