# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Generation strategies, selected per request by ``context["strategy"]``.

Each strategy turns a natural-language question into an OpenSearch ``_search``
body; the agent (``agent.py``) owns the shared plumbing (cluster client, model,
credentials, fallback) and dispatches here. To add one — e.g. search-template
fill — write its module and register it in ``STRATEGIES``; nothing else changes.
"""

from __future__ import annotations

from agents.agentic_search.strategies.direct_dsl import DirectDslStrategy
from agents.agentic_search.strategies.multi_template_fill import (
    MultiTemplateFillStrategy,
)
from agents.agentic_search.strategies.template_fill import TemplateFillStrategy

# Registry keyed by strategy name. `direct_dsl` is the default when a request omits
# `context.strategy`. The template strategies fill a search template's params instead of
# authoring DSL; the agent picks between them from how many distinct template ids a
# request carries (see AgenticSearchAgent._select_strategy), and both degrade to
# `direct_dsl`.
STRATEGIES = {
    DirectDslStrategy.name: DirectDslStrategy(),
    TemplateFillStrategy.name: TemplateFillStrategy(),
    MultiTemplateFillStrategy.name: MultiTemplateFillStrategy(),
}
DEFAULT_STRATEGY = DirectDslStrategy.name
# One template to fill: the dedicated single-template path.
SINGLE_TEMPLATE_STRATEGY = TemplateFillStrategy.name
# Several candidates: one call chooses among them and fills the winner.
MULTI_TEMPLATE_STRATEGY = MultiTemplateFillStrategy.name
