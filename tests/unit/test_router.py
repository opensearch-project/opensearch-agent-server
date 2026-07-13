"""Unit tests for page-context routing."""

from __future__ import annotations

import pytest

from orchestrator.registry import AgentRegistration, AgentRegistry
from orchestrator.router import PageContextRouter

pytestmark = pytest.mark.unit


@pytest.fixture
def registered_agents() -> tuple[AgentRegistry, AgentRegistration, AgentRegistration]:
    registry = AgentRegistry()
    fallback = AgentRegistration(
        name="default",
        description="Fallback agent",
        is_default=True,
    )
    discover = AgentRegistration(
        name="discover",
        description="Discover page agent",
        page_contexts=["discover"],
    )
    registry.register(fallback)
    registry.register(discover)
    return registry, fallback, discover


def test_exact_page_context_routes_to_registered_agent(registered_agents):
    registry, _, discover = registered_agents

    assert PageContextRouter(registry).route("discover") is discover


def test_unknown_page_context_routes_to_fallback(registered_agents):
    registry, fallback, _ = registered_agents

    assert PageContextRouter(registry).route("unknown") is fallback


def test_none_page_context_routes_to_fallback(registered_agents):
    registry, fallback, _ = registered_agents

    assert PageContextRouter(registry).route(None) is fallback


def test_empty_page_context_routes_to_fallback(registered_agents):
    registry, fallback, _ = registered_agents

    assert PageContextRouter(registry).route("") is fallback


def test_missing_fallback_raises_runtime_error():
    registry = AgentRegistry()
    registry.register(
        AgentRegistration(
            name="discover",
            description="Discover page agent",
            page_contexts=["discover"],
        )
    )

    with pytest.raises(RuntimeError, match="No default agent registered"):
        PageContextRouter(registry).route("unknown")


def test_page_context_matching_is_case_sensitive(registered_agents):
    registry, fallback, _ = registered_agents

    assert PageContextRouter(registry).route("Discover") is fallback
