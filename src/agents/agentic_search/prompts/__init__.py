"""Prompts for the agentic-search agent, split by generation strategy.

- :mod:`.direct_dsl` — rules, examples, ``EmitSearch`` schema, and cached system
  blocks for free-DSL generation (the default strategy).

Strategies import their own prompt set from the submodule directly. Only
``FALLBACK_DSL`` is re-exported here, since the agent shell returns it on any
generation failure regardless of strategy.
"""

from __future__ import annotations

from agents.agentic_search.prompts.direct_dsl import FALLBACK_DSL

__all__ = ["FALLBACK_DSL"]
