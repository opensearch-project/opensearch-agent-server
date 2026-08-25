# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for orchestrator.registry.AgentRegistry."""

from __future__ import annotations

import pytest

from orchestrator.registry import AgentRegistration, AgentRegistry

pytestmark = pytest.mark.unit


class TestAgentRegistration:
    def test_defaults(self):
        reg = AgentRegistration(name="a", description="d")
        assert reg.page_contexts == []
        assert reg.is_default is False


class TestAgentRegistry:
    def test_register_and_lookup_by_context(self):
        registry = AgentRegistry()
        reg = AgentRegistration(
            name="art", description="ART", page_contexts=["search-relevance"]
        )
        registry.register(reg)
        assert registry.get_agent_for_context("search-relevance") is reg

    def test_lookup_unknown_context_returns_none(self):
        registry = AgentRegistry()
        assert registry.get_agent_for_context("nope") is None

    def test_register_default(self):
        registry = AgentRegistry()
        reg = AgentRegistration(name="default", description="D", is_default=True)
        registry.register(reg)
        assert registry.get_default() is reg

    def test_get_default_when_none_registered(self):
        registry = AgentRegistry()
        assert registry.get_default() is None

    def test_duplicate_name_raises(self):
        registry = AgentRegistry()
        registry.register(AgentRegistration(name="dup", description="one"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(AgentRegistration(name="dup", description="two"))

    def test_duplicate_page_context_raises(self):
        registry = AgentRegistry()
        registry.register(
            AgentRegistration(name="a", description="A", page_contexts=["ctx"])
        )
        with pytest.raises(ValueError, match="already mapped"):
            registry.register(
                AgentRegistration(name="b", description="B", page_contexts=["ctx"])
            )

    def test_list_agents(self):
        registry = AgentRegistry()
        registry.register(AgentRegistration(name="a", description="A"))
        registry.register(AgentRegistration(name="b", description="B"))
        names = {r.name for r in registry.list_agents()}
        assert names == {"a", "b"}
