"""Unit tests for the direct_dsl strategy's provider routing.

The forced-tool path is Bedrock-specific (drives ``converse_stream`` directly),
so the strategy must fall back to the portable strands ``structured_output`` path
for other providers (e.g. Ollama). These tests stub both paths — nothing hits a
cluster or an LLM — and assert the right one is chosen per provider.
"""

from __future__ import annotations

import pytest

from agents.agentic_search.strategies import direct_dsl as direct_dsl_module
from agents.agentic_search.strategies.direct_dsl import DirectDslStrategy
from agents.agentic_search.strategies.forced_tool import supports_forced_tool

pytestmark = pytest.mark.unit


class _BedrockLikeModel:
    """Has a ``client`` exposing ``converse_stream`` (the forced-tool capability)."""

    class _Client:
        def converse_stream(self, **kwargs):  # pragma: no cover - never called (stubbed)
            raise AssertionError("real converse_stream should be stubbed out")

    client = _Client()
    config = {"model_id": "bedrock-model"}


class _OllamaLikeModel:
    """No ``converse_stream`` on its client — must use the portable path."""

    class _Client:
        pass

    client = _Client()
    config = {"model_id": "ollama-model"}


class _StubClient:
    """OpenSearch client stub for the sample-document fetch."""

    def search(self, index, body, _source):
        return {"hits": {"hits": [{"_source": {"title": "Widget", "color": "red"}}]}}


class _Request:
    def __init__(self, model):
        self.model = model
        self.client = _StubClient()
        self.question = "red widgets"
        self.index_name = "products"
        self.mapping = '{"properties": {"color": {"type": "keyword"}}}'
        self.context: dict = {}


class _Result:
    reason = "match color=red"
    dsl = {"query": {"term": {"color": "red"}}}


def test_supports_forced_tool_detects_bedrock_vs_ollama():
    assert supports_forced_tool(_BedrockLikeModel()) is True
    assert supports_forced_tool(_OllamaLikeModel()) is False
    assert supports_forced_tool(object()) is False  # no client attribute


def test_bedrock_uses_forced_tool_path(monkeypatch):
    calls = {}

    def _fake_forced(*, model, schema_model, system_blocks, user_message):
        calls["forced"] = True
        return _Result()

    def _fail_agent(*a, **k):  # the strands Agent path must NOT be used on Bedrock
        raise AssertionError("Bedrock should use forced_tool_fill, not Agent")

    monkeypatch.setattr(direct_dsl_module, "forced_tool_fill", _fake_forced)
    monkeypatch.setattr(direct_dsl_module, "Agent", _fail_agent)

    dsl = DirectDslStrategy().generate(_Request(_BedrockLikeModel()))

    assert calls.get("forced") is True
    assert dsl == {"query": {"term": {"color": "red"}}}


def test_ollama_falls_back_to_structured_output(monkeypatch):
    """The core review concern: a non-Bedrock model must not crash into fallback."""
    calls = {}

    def _fail_forced(**k):
        raise AssertionError("Ollama should not use the Bedrock forced-tool path")

    class _FakeAgent:
        def __init__(self, *a, **k):
            calls["agent_built"] = True

        def __call__(self, user_msg, structured_output_model):
            calls["structured_output"] = True
            return type("R", (), {"structured_output": _Result()})()

    monkeypatch.setattr(direct_dsl_module, "forced_tool_fill", _fail_forced)
    monkeypatch.setattr(direct_dsl_module, "Agent", _FakeAgent)

    dsl = DirectDslStrategy().generate(_Request(_OllamaLikeModel()))

    assert calls.get("structured_output") is True
    assert dsl == {"query": {"term": {"color": "red"}}}
