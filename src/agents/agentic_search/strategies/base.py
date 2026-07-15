"""Generation-strategy seam for the agentic-search agent.

A strategy turns a natural-language question into an OpenSearch ``_search``
body (a dict); the agent owns the shared plumbing (cluster client, model,
fallback) and picks the strategy from ``context["strategy"]``. New generation
methods (e.g. search-template fill) are new modules here, listed in
``strategies/__init__.py``. Anything whose output is not a ``_search`` body
(e.g. a PPL generator) should be a sibling agent, not a strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from opensearchpy import OpenSearch


@dataclass(frozen=True)
class GenerationRequest:
    """Inputs a strategy gets: the NLQ, the pre-fetched mapping (JSON string),
    the full request ``context`` for strategy-specific inputs, the shared model,
    and a request-scoped cluster client carrying the caller's credentials."""

    question: str
    index_name: str
    mapping: str
    context: dict[str, Any]
    model: Any
    client: OpenSearch


class GenerationStrategy(Protocol):
    """Returns a ``_search`` body dict; raising degrades to the fallback."""

    name: str

    def generate(self, request: GenerationRequest) -> dict[str, Any]: ...
