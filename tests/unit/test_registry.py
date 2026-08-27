"""Unit tests for the AgentRegistry — the page-context-to-agent mapping.

``src/orchestrator/registry.py`` maps *page contexts* (the part of the UI a
request originates from, e.g. ``"search-relevance"``) to *agent registrations*.
The orchestrator/router consults this registry at request time to decide which
sub-agent should handle an incoming request.

These tests verify:
1. Registering an agent and retrieving it by page context.
2. Registering multiple agents with different page contexts.
3. Duplicate name / duplicate page-context registration raises ``ValueError``.
4. ``list_agents()`` returns every registered agent.
5. Default agent resolution via ``is_default=True`` / ``get_default()``.
6. ``get_default()`` returns ``None`` when no default is registered.
7. An agent with an empty ``page_contexts`` list registers cleanly.
"""

from __future__ import annotations

import pytest

from orchestrator.registry import AgentRegistration, AgentRegistry

pytestmark = pytest.mark.unit


def _registration(
    name: str,
    *,
    page_contexts: list[str] | None = None,
    is_default: bool = False,
    description: str = "",
) -> AgentRegistration:
    """Build an AgentRegistration with sensible test defaults."""
    return AgentRegistration(
        name=name,
        description=description or f"{name} agent",
        page_contexts=page_contexts if page_contexts is not None else [],
        is_default=is_default,
    )


# Register + retrieve by page context


def test_register_and_retrieve_by_page_context() -> None:
    registry = AgentRegistry()
    reg = _registration("search-agent", page_contexts=["search-relevance"])

    registry.register(reg)

    resolved = registry.get_agent_for_context("search-relevance")
    assert resolved is reg
    assert resolved.name == "search-agent"


def test_get_agent_for_unknown_context_on_empty_registry_returns_none() -> None:
    registry = AgentRegistry()

    assert registry.get_agent_for_context("does-not-exist") is None


def test_get_agent_for_unknown_context_returns_none() -> None:
    registry = AgentRegistry()
    registry.register(_registration("search-agent", page_contexts=["search-relevance"]))

    assert registry.get_agent_for_context("does-not-exist") is None


# Multiple agents with different page contexts


def test_register_multiple_agents_with_different_contexts() -> None:
    registry = AgentRegistry()
    search = _registration("search-agent", page_contexts=["search-relevance"])
    obs = _registration("obs-agent", page_contexts=["observability", "alerting"])

    registry.register(search)
    registry.register(obs)

    assert registry.get_agent_for_context("search-relevance") is search
    assert registry.get_agent_for_context("observability") is obs
    assert registry.get_agent_for_context("alerting") is obs


def test_duplicate_agent_name_raises() -> None:
    registry = AgentRegistry()
    registry.register(_registration("dup", page_contexts=["ctx-a"]))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_registration("dup", page_contexts=["ctx-b"]))


def test_duplicate_page_context_raises() -> None:
    registry = AgentRegistry()
    registry.register(_registration("agent-a", page_contexts=["shared-ctx"]))

    with pytest.raises(ValueError, match="already mapped"):
        registry.register(_registration("agent-b", page_contexts=["shared-ctx"]))


def test_duplicate_page_context_does_not_partially_register() -> None:
    """A conflicting registration must not leave the second agent half-added."""
    registry = AgentRegistry()
    registry.register(_registration("agent-a", page_contexts=["shared-ctx"]))

    with pytest.raises(ValueError):
        registry.register(
            _registration("agent-b", page_contexts=["fresh-ctx", "shared-ctx"])
        )

    # agent-b must not be in the registry, and its first (non-conflicting)
    # context must not have leaked into the map before the conflict was hit.
    assert all(a.name != "agent-b" for a in registry.list_agents())
    # verify that "fresh-ctx" (the non-conflicting context registered before the conflict was hit) wasn't
    # written into the page-context 
    assert registry.get_agent_for_context("fresh-ctx") is None


# list_agents()


def test_list_agents_returns_all_registered() -> None:
    registry = AgentRegistry()
    assert registry.list_agents() == []

    a = _registration("a", page_contexts=["ctx-a"])
    b = _registration("b", page_contexts=["ctx-b"])
    registry.register(a)
    registry.register(b)

    listed = registry.list_agents()
    assert len(listed) == 2
    assert {x.name for x in listed} == {"a", "b"}


# Default agent resolution


def test_get_default_returns_default_agent() -> None:
    registry = AgentRegistry()
    normal = _registration("normal", page_contexts=["ctx-a"])
    default = _registration("fallback", page_contexts=["ctx-b"], is_default=True)

    registry.register(normal)
    registry.register(default)

    assert registry.get_default() is default


def test_get_default_returns_none_when_no_default_registered() -> None:
    registry = AgentRegistry()
    registry.register(_registration("normal", page_contexts=["ctx-a"]))

    assert registry.get_default() is None


# Empty page_contexts


def test_agent_with_empty_page_contexts_registers_cleanly() -> None:
    registry = AgentRegistry()
    reg = _registration("contextless", page_contexts=[])

    registry.register(reg)

    # It is listed and resolvable as default-only, but not via any context.
    assert reg in registry.list_agents()
    assert registry.get_agent_for_context("anything") is None


def test_default_agent_with_empty_page_contexts() -> None:
    registry = AgentRegistry()
    reg = _registration("contextless-default", page_contexts=[], is_default=True)

    registry.register(reg)

    assert registry.get_default() is reg
