# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Plumbing shared by the search-template strategies.

Both template strategies read the same context keys, render through the cluster's own
Mustache engine, and degrade to the free-DSL generator on any failure. That shared tail
lives here so the two strategies — and the agent's routing gate, which must count
candidates the same way a strategy resolves them — cannot drift apart.

A strategy subclasses :class:`TemplateStrategyBase` for the constructor, the render step
and the fallback step, and supplies its own ``name`` and ``generate``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from agents.agentic_search.strategies.base import GenerationRequest, GenerationStrategy
from agents.agentic_search.strategies.direct_dsl import DirectDslStrategy
from agents.agentic_search.template_schema import TemplateSchemaCache

logger = logging.getLogger(__name__)

# Context keys naming the search template(s) to fill.
TEMPLATE_ID_KEY = "template_id"
TEMPLATE_IDS_KEY = "template_ids"


def distinct_template_ids(context: dict[str, Any]) -> list[str]:
    """Return the distinct template ids a request asks for, in order.

    Accepts a single id under either key, so a caller may send a one-element
    ``template_ids`` list without changing behavior. Duplicates are collapsed because
    they do not represent a real choice.

    Shared by the agent's routing gate and the multi-template strategy: the number of
    candidates the gate counts and the list the strategy resolves must never disagree.
    """
    raw = context.get(TEMPLATE_IDS_KEY)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raw = []
    ids = [str(x) for x in raw if x]
    one = context.get(TEMPLATE_ID_KEY)
    if one:
        ids.append(str(one))
    seen: set[str] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


class TemplateStrategyBase:
    """Constructor, render step and free-DSL fallback shared by template strategies.

    ``needs_mapping = False`` tells the agent to skip the per-query mapping fetch on the
    happy path (filling typed params uses no mapping); :meth:`_fallback_generate` fetches
    it lazily on the only path that needs it.
    """

    # The fill prompts carry no index mapping (the win is output tokens, §7), so the
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


__all__ = [
    "TEMPLATE_ID_KEY",
    "TEMPLATE_IDS_KEY",
    "TemplateStrategyBase",
    "distinct_template_ids",
]
