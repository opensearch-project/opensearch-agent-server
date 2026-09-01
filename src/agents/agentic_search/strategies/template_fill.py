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

import logging
from typing import Any

from agents.agentic_search.prompts.template_fill import (
    FILL_SYSTEM_BLOCKS,
    FILL_USER_PROMPT,
)
from agents.agentic_search.strategies.base import GenerationRequest
from agents.agentic_search.strategies.forced_tool import forced_tool_fill
from agents.agentic_search.strategies.template_base import (
    TEMPLATE_ID_KEY,
    TEMPLATE_IDS_KEY,
    TemplateStrategyBase,
)
from agents.agentic_search.template_schema import CANNOT_EXPRESS_FIELD

logger = logging.getLogger(__name__)


class _TemplateCannotExpress(Exception):
    """The model abstained: the question needs a capability this template lacks.

    Raised on the happy path so :meth:`TemplateFillStrategy.generate` routes to the
    free-DSL fallback — the same handling as a structural failure, but triggered by the
    model's own judgment rather than a broken render (§6).
    """


class TemplateFillStrategy(TemplateStrategyBase):
    """Fill a search template's params and render it into a ``_search`` body.

    On any failure the request degrades to ``fallback`` (the ported free-DSL
    generator). The constructor, the render step, the free-DSL fallback and the
    inherited ``needs_mapping = False`` all come from
    :class:`~agents.agentic_search.strategies.template_base.TemplateStrategyBase`.
    """

    name = "template_fill"

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        template_id = self._template_id(request.context)
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

    @staticmethod
    def _template_id(context: dict[str, Any]) -> str | None:
        """Return the single template to fill, from either context key.

        A caller may name the template as a scalar ``template_id`` or as a one-element
        ``template_ids`` list; both mean "fill this one template". A list carrying
        several ids is routed to the multi-template strategy before reaching here, so
        only its first entry is honored if one somehow arrives.

        Deliberately *not* folded into
        :func:`~agents.agentic_search.strategies.template_base.distinct_template_ids`:
        that helper orders ``template_ids`` first and appends the scalar, while this
        gives the scalar precedence. The orders disagree when a caller sends both keys
        with different values and pins ``context.strategy`` to this strategy (which
        bypasses the candidate-count gate), and taking the other order would silently
        fill a different template.
        """
        one = context.get(TEMPLATE_ID_KEY)
        if one:
            return str(one)
        many = context.get(TEMPLATE_IDS_KEY)
        if isinstance(many, str):
            return many or None
        if isinstance(many, list):
            for candidate in many:
                if candidate:
                    return str(candidate)
        return None

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
