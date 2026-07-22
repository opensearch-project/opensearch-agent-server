"""Generation strategies, selected per request by ``context["strategy"]``.

Each strategy turns a natural-language question into an OpenSearch ``_search``
body; the agent (``agent.py``) owns the shared plumbing (cluster client, model,
credentials, fallback) and dispatches here. To add one — e.g. search-template
fill — write its module and register it in ``STRATEGIES``; nothing else changes.
"""

from __future__ import annotations

from agents.agentic_search.strategies.direct_dsl import DirectDslStrategy

# Registry keyed by strategy name. `direct_dsl` is the default when a request
# omits `context.strategy`.
STRATEGIES = {DirectDslStrategy.name: DirectDslStrategy()}
DEFAULT_STRATEGY = DirectDslStrategy.name
