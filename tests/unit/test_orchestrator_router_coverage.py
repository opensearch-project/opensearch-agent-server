# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for orchestrator.router.PageContextRouter."""

from __future__ import annotations

import pytest

from orchestrator.registry import AgentRegistration, AgentRegistry
from orchestrator.router import PageContextRouter

pytestmark = pytest.mark.unit


def _registry_with_default() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentRegistration(
            name="art", description="ART", page_contexts=["search-relevance"]
        )
    )
    registry.register(
        AgentRegistration(name="olly", description="Olly", is_default=True)
    )
    return registry


class TestPageContextRouter:
    def test_tier1_context_match(self):
        router = PageContextRouter(_registry_with_default())
        agent = router.route("search-relevance")
        assert agent.name == "art"

    def test_falls_back_to_default_when_no_match(self):
        router = PageContextRouter(_registry_with_default())
        agent = router.route("unknown-context")
        assert agent.name == "olly"

    def test_falls_back_to_default_when_context_none(self):
        router = PageContextRouter(_registry_with_default())
        agent = router.route(None)
        assert agent.name == "olly"

    def test_raises_when_no_default_and_no_match(self):
        registry = AgentRegistry()
        registry.register(
            AgentRegistration(name="art", description="ART", page_contexts=["sr"])
        )
        router = PageContextRouter(registry)
        with pytest.raises(RuntimeError, match="No default agent registered"):
            router.route("no-match")
