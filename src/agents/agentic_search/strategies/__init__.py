"""Generation strategies, keyed by the ``context["strategy"]`` request field.

To add one: create its module, add it to STRATEGIES. Nothing else changes.
"""

from __future__ import annotations

from agents.agentic_search.strategies.base import GenerationRequest, GenerationStrategy
from agents.agentic_search.strategies.direct_dsl import DirectDslStrategy

STRATEGIES: dict[str, GenerationStrategy] = {
    strategy.name: strategy for strategy in (DirectDslStrategy(),)
}
DEFAULT_STRATEGY = DirectDslStrategy.name

__all__ = [
    "DEFAULT_STRATEGY",
    "STRATEGIES",
    "GenerationRequest",
    "GenerationStrategy",
]
