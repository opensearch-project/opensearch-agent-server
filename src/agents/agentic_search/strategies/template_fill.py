"""Search-template fill: the model fills a template's params, OpenSearch renders.

This is design part 2, layered on the free-DSL port (G0). In template mode the LLM
does one thing — fill *this template's* Mustache parameters — and OpenSearch's own
engine renders them into the stored body. The model never authors DSL, so it emits
~5-30 tokens (values only) instead of ~300 (a decode-latency win, G1), and the
output is structurally valid by construction because the template owns the query
shape (G3).

Steps (§4.7):
  1. Resolve the per-template ``FillTemplate`` model from the cached param-schema.
  2. One forced tool call fills it (reusing the port's forced-toolChoice mechanism).
  3. Drop unset params, hand the values to ``POST _render/template``, and unwrap the
     rendered ``_search`` body.

Reliability (§6): a single fallback. If anything above fails — unregistered
template, a fill that won't validate, a body that won't render to legal DSL — we
degrade to the ported free-DSL generator rather than return a wrong or empty
result. Free-DSL is the proven path and carries its own ``match_all`` fail-safe, so
we don't re-implement one here. Escaping is delegated to OpenSearch's renderer (the
whole point of choosing ``_render/template`` in D4), not reimplemented in Python.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from agents.agentic_search.prompts.template_fill import (
    FILL_SYSTEM_BLOCKS,
    FILL_USER_PROMPT,
)
from agents.agentic_search.strategies.base import GenerationRequest, GenerationStrategy
from agents.agentic_search.strategies.direct_dsl import DirectDslStrategy
from agents.agentic_search.strategies.forced_tool import forced_tool_fill
from agents.agentic_search.template_schema import (
    CANNOT_EXPRESS_FIELD,
    TemplateSchemaCache,
)

logger = logging.getLogger(__name__)

TEMPLATE_ID_KEY = "template_id"


class _TemplateCannotExpress(Exception):
    """The model abstained: the question needs a capability this template lacks.

    Raised on the happy path so :meth:`TemplateFillStrategy.generate` routes to the
    free-DSL fallback — the same handling as a structural failure, but triggered by the
    model's own judgment rather than a broken render (§6).
    """


class TemplateFillStrategy:
    """Fill a search template's params and render it into a ``_search`` body.

    On any failure the request degrades to ``fallback`` (the ported free-DSL
    generator). ``needs_mapping = False`` tells the agent to skip the per-query
    mapping fetch on the happy path (fill doesn't use the mapping); the fallback
    branch fetches it lazily only when it's actually needed.
    """

    name = "template_fill"
    # The fill prompt carries no index mapping (the win is output tokens, §7), so the
    # agent can skip the mapping fetch upfront; the fallback re-adds it on demand.
    needs_mapping = False

    def __init__(
        self,
        *,
        fallback: GenerationStrategy | None = None,
        schema_cache: TemplateSchemaCache | None = None,
    ) -> None:
        # Free-DSL fallback (G0). Defaults to the standard direct-DSL generator; a
        # benchmark harness may inject a forced variant instead.
        self._fallback = fallback if fallback is not None else DirectDslStrategy()
        self._schema_cache = (
            schema_cache if schema_cache is not None else TemplateSchemaCache()
        )

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        template_id = request.context.get(TEMPLATE_ID_KEY)
        if not template_id:
            # template_fill was selected without a template_id — nothing to fill.
            # Degrade rather than raise so the request still returns results.
            logger.warning(
                "template_fill selected but no template_id in context; falling back"
            )
            return self._fallback_generate(request)

        try:
            return self._fill_and_render(request, template_id)
        except _TemplateCannotExpress:
            # Not a failure: the model judged the question outside the template's
            # expressive range and asked for the free-DSL path. Expected and healthy.
            logger.info(
                "Template %s cannot express the question; routing to free-DSL",
                template_id,
            )
            return self._fallback_generate(request)
        except Exception as e:  # noqa: BLE001 - any fill/render failure degrades to free-DSL
            logger.warning(
                "Template fill failed for template_id=%s (%s); falling back to free-DSL",
                template_id,
                e,
            )
            return self._fallback_generate(request)

    def _fill_and_render(
        self, request: GenerationRequest, template_id: str
    ) -> dict[str, Any]:
        """The happy path: fill the template's params and render them into DSL.

        Raises on any failure so :meth:`generate` can degrade to the fallback.
        """
        schema = self._schema_cache.get(template_id, request.client)

        filled = forced_tool_fill(
            model=request.model,
            schema_model=schema.fill_model,
            system_blocks=FILL_SYSTEM_BLOCKS,
            user_message=FILL_USER_PROMPT.format(question=request.question),
        )
        # Only the params the model filled: dropping unset optionals lets the body's
        # inverted-section defaults ({{^size}}10{{/size}}) and optional clauses
        # ({{#color}}...{{/color}}) work, where a null would render an empty slot and
        # break the JSON. ``by_alias`` recovers the real names from sanitized fields.
        params = filled.model_dump(by_alias=True, exclude_none=True)

        # Escape hatch: the synthetic abstain field (never a real Mustache param) lets
        # the model decline a question this template can't express, instead of forcing a
        # fill that renders valid-but-wrong DSL. Strip it either way — it must not reach
        # the renderer — and route an abstention to the free-DSL fallback (§6).
        abstained = bool(params.pop(CANNOT_EXPRESS_FIELD, False))
        if abstained:
            raise _TemplateCannotExpress(template_id)

        rendered = self._render(request.client, template_id, params)
        logger.info(
            "Template fill for template_id=%s rendered %d params",
            template_id,
            len(params),
        )
        return rendered

    @staticmethod
    def _render(
        client: Any, template_id: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Render the stored body via ``POST _render/template`` and unwrap the DSL.

        OpenSearch's own Mustache engine renders (and safety-checks) the query, so
        JSON-escaping and dialect are the cluster's concern, not ours (D4). The
        rendered ``_search`` body comes back under ``template_output``.

        Raises:
            ValueError: The response has no ``template_output`` or it isn't a
                ``_search`` body object.
        """
        resp = client.render_search_template(id=template_id, body={"params": params})
        output = resp.get("template_output") if isinstance(resp, dict) else None
        if output is None:
            raise ValueError("_render/template returned no template_output")
        # Most opensearch-py versions parse template_output into a dict; tolerate a
        # string form by parsing it (still using the cluster's rendered bytes).
        if isinstance(output, str):
            output = json.loads(output)
        if not isinstance(output, dict):
            raise ValueError("rendered template_output is not a _search body object")
        return output

    def _fallback_generate(self, request: GenerationRequest) -> dict[str, Any]:
        """Run the free-DSL fallback, re-adding the mapping the happy path skipped.

        Because ``needs_mapping = False`` the agent didn't fetch the mapping, but the
        free-DSL generator needs it — so fetch it here (only on the fallback path)
        and hand the fallback a request that carries it.
        """
        req = request
        if not request.mapping:
            try:
                mapping = json.dumps(
                    request.client.indices.get_mapping(index=request.index_name)
                )
                req = replace(request, mapping=mapping)
            except Exception as e:  # noqa: BLE001 - let the fallback try with what it has
                logger.warning(
                    "Fallback mapping fetch failed for index=%s (%s)",
                    request.index_name,
                    e,
                )
        return self._fallback.generate(req)
