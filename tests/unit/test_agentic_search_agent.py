"""Unit tests for the agentic_search agent (reached via POST /invoke).

The OpenSearch client and the generation strategy are stubbed, so nothing hits a
cluster or an LLM. Covers: strategy dispatch and mapping/token plumbing, the
guard paths (bad query / missing index / unknown strategy / non-dict result),
and that every failure degrades to FALLBACK_DSL.
"""

from __future__ import annotations

import json

import pytest

from agents.agentic_search import agent as agent_module
from agents.agentic_search.agent import AgenticSearchAgent
from agents.agentic_search.prompts import FALLBACK_DSL

pytestmark = pytest.mark.unit

MATCH_ALL = {"size": 10, "query": {"match_all": {}}}


class _StubStrategy:
    """Records the generate() call and returns a canned body (or raises)."""

    name = "direct_dsl"

    def __init__(self, *, returns=None, raises=None):
        self._returns = returns if returns is not None else {"query": {"match_all": {}}}
        self._raises = raises
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        if self._raises is not None:
            raise self._raises
        return self._returns


class _StubClient:
    """Stands in for the OpenSearch client; records the mapping fetch."""

    def __init__(self, mapping=None):
        self._mapping = (
            mapping if mapping is not None else {"products": {"mappings": {}}}
        )
        self.fetched = []

    class _Indices:
        def __init__(self, outer):
            self._outer = outer

        def get_mapping(self, index):
            self._outer.fetched.append(index)
            return self._outer._mapping

    @property
    def indices(self):
        return _StubClient._Indices(self)


@pytest.fixture
def make_agent(monkeypatch):
    """Build an AgenticSearchAgent with model creation + client stubbed out."""

    def _factory(strategy=None, client=None):
        monkeypatch.setattr(agent_module, "create_model", lambda: object())
        strat = strategy or _StubStrategy()
        monkeypatch.setattr(agent_module, "STRATEGIES", {strat.name: strat})
        monkeypatch.setattr(agent_module, "DEFAULT_STRATEGY", strat.name)
        a = AgenticSearchAgent("http://localhost:9200")
        monkeypatch.setattr(a, "_client", lambda auth_token: client or _StubClient())
        return a, strat

    return _factory


def test_happy_path_dispatches_and_returns_dsl(make_agent):
    strat = _StubStrategy(returns={"query": {"term": {"color": "red"}}})
    client = _StubClient(mapping={"products": {"mappings": {"properties": {}}}})
    a, strat = make_agent(strategy=strat, client=client)

    out = a("red shoes", context={"index_name": "products"}, auth_token="tok")

    assert json.loads(out) == {"query": {"term": {"color": "red"}}}
    # mapping fetched for the requested index; strategy got a GenerationRequest
    # carrying the NLQ, mapping, context, model, and client.
    assert client.fetched == ["products"]
    req = strat.calls[0]
    assert req.question == "red shoes" and req.index_name == "products"
    assert json.loads(req.mapping) == {"products": {"mappings": {"properties": {}}}}
    assert req.context == {"index_name": "products"}
    assert req.client is client and req.model is not None


def test_declares_context_awareness():
    # The orchestrator keys context/auth forwarding off this flag.
    assert AgenticSearchAgent.accepts_invoke_context is True


def test_missing_index_name_falls_back(make_agent):
    a, strat = make_agent()
    assert a("red shoes", context={}) == FALLBACK_DSL
    assert strat.calls == []  # guarded before reaching the strategy


def test_empty_query_falls_back(make_agent):
    a, strat = make_agent()
    assert a("   ", context={"index_name": "products"}) == FALLBACK_DSL
    assert strat.calls == []


def test_non_string_query_falls_back(make_agent):
    a, strat = make_agent()
    assert a(["not", "a", "string"], context={"index_name": "products"}) == FALLBACK_DSL
    assert strat.calls == []


def test_non_dict_context_falls_back(make_agent):
    a, strat = make_agent()
    assert a("red shoes", context="oops") == FALLBACK_DSL
    assert strat.calls == []


def test_unknown_strategy_falls_back(make_agent):
    a, strat = make_agent()
    out = a("red shoes", context={"index_name": "products", "strategy": "nope"})
    assert out == FALLBACK_DSL
    assert strat.calls == []


def test_strategy_exception_falls_back(make_agent):
    a, strat = make_agent(strategy=_StubStrategy(raises=RuntimeError("bedrock down")))
    assert a("red shoes", context={"index_name": "products"}) == FALLBACK_DSL


def test_non_object_strategy_result_falls_back(make_agent):
    # A strategy must return a _search body object; anything else degrades.
    a, strat = make_agent(strategy=_StubStrategy(returns=["not", "an", "object"]))
    assert a("red shoes", context={"index_name": "products"}) == FALLBACK_DSL
