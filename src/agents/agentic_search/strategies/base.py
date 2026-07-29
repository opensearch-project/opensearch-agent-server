"""The generation-strategy contract for the agentic-search agent.

A strategy turns a natural-language question into an OpenSearch ``_search`` body
(a dict). The agent owns the shared plumbing (cluster client, model, credentials,
fallback) and passes it to the strategy as a :class:`GenerationRequest`; the
strategy is selected per request by ``context["strategy"]``. New generation
methods (e.g. search-template fill) are new modules registered in
``strategies/__init__.py`` — adding one means adding code, not editing this
contract. A generator whose output is not a ``_search`` body (e.g. a PPL or
intermediate-representation generator) belongs in a sibling agent, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from opensearchpy import OpenSearch


@dataclass(frozen=True)
class GenerationRequest:
    """Everything a strategy needs to generate a ``_search`` body.

    Bundled so a new shared input can be added here without touching every
    strategy's signature. Strategy-specific inputs travel in ``context``.
    """

    question: str
    index_name: str
    mapping: str
    context: dict[str, Any]
    model: Any
    client: OpenSearch


class GenerationStrategy(Protocol):
    """Returns a ``_search`` body dict; raising degrades the request to the fallback."""

    name: str

    def generate(self, request: GenerationRequest) -> dict[str, Any]: ...
