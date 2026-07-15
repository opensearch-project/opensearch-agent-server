"""Unit tests for the ``agentic_search`` agent (reached via POST /invoke).

Mocks the strands Agent, ``create_model``, and opensearch-py so the agent's
wiring is verified with no cluster, no AWS, and no LLM:
1. A successful generation fetches the mapping, forces an EmitSearch structured
   call, and returns the DSL as a JSON string.
2. The caller's bearer token and the configured verify_certs reach the client.
3. Bad input, unknown strategy, generation errors, and non-object strategy
   output all degrade to the fallback match_all DSL.
4. Regression guards for the two load-bearing, easily-dropped details: the
   trailing Bedrock cache point and the ``structured_output_model`` call.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.agentic_search.agent import AgenticSearchAgent
from agents.agentic_search.prompts import FALLBACK_DSL, SYSTEM_BLOCKS, EmitSearch
from agents.agentic_search.strategies import STRATEGIES

pytestmark = pytest.mark.unit

MATCH_ALL = {"size": 10, "query": {"match_all": {}}}


def _emit(dsl: dict) -> EmitSearch:
    return EmitSearch(reason="r", dsl=dsl)


class TestGeneration:
    """Successful direct_dsl generation and the request wiring."""

    @patch("agents.agentic_search.strategies.direct_dsl.Agent")
    @patch("agents.agentic_search.agent.create_model")
    @patch("agents.agentic_search.agent.OpenSearch")
    def test_reads_mapping_and_returns_dsl_string(self, mock_os, mock_model, mock_agent):
        mock_os.return_value.indices.get_mapping.return_value = {
            "idx": {"mappings": {"properties": {"status": {"type": "keyword"}}}}
        }
        dsl_obj = {"size": 10, "query": {"term": {"status": "active"}}}
        mock_agent.return_value.return_value.structured_output = _emit(dsl_obj)

        out = AgenticSearchAgent("https://localhost:9200")(
            "active items", context={"index_name": "idx"}, auth_token="tok"
        )

        # Returns the dsl serialized to a STRING (what the passthrough needs).
        assert isinstance(out, str)
        assert json.loads(out) == dsl_obj
        # Mapping read for the right index and fed into the prompt.
        mock_os.return_value.indices.get_mapping.assert_called_once_with(index="idx")
        prompt_arg = mock_agent.return_value.call_args[0][0]
        assert "active items" in prompt_arg and "status" in prompt_arg

    @patch("agents.agentic_search.strategies.direct_dsl.Agent")
    @patch("agents.agentic_search.agent.create_model")
    @patch("agents.agentic_search.agent.OpenSearch")
    def test_bearer_token_and_verify_certs_forwarded(self, mock_os, mock_model, mock_agent):
        mock_os.return_value.indices.get_mapping.return_value = {"idx": {}}
        mock_agent.return_value.return_value.structured_output = _emit(MATCH_ALL)

        AgenticSearchAgent("https://localhost:9200", verify_certs=True)(
            "q", context={"index_name": "idx"}, auth_token="tok"
        )

        _, kwargs = mock_os.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer tok"}
        assert kwargs["verify_certs"] is True

    @patch("agents.agentic_search.strategies.direct_dsl.Agent")
    @patch("agents.agentic_search.agent.create_model")
    @patch("agents.agentic_search.agent.OpenSearch")
    def test_no_token_means_no_auth_header(self, mock_os, mock_model, mock_agent):
        mock_os.return_value.indices.get_mapping.return_value = {"idx": {}}
        mock_agent.return_value.return_value.structured_output = _emit(MATCH_ALL)

        AgenticSearchAgent("https://localhost:9200")("q", context={"index_name": "idx"})

        _, kwargs = mock_os.call_args
        assert kwargs["headers"] is None
        assert kwargs["verify_certs"] is False


class TestFallback:
    """Every failure mode degrades to FALLBACK_DSL rather than raising."""

    @patch("agents.agentic_search.strategies.direct_dsl.Agent")
    @patch("agents.agentic_search.agent.create_model")
    @patch("agents.agentic_search.agent.OpenSearch")
    def test_generation_error_falls_back(self, mock_os, mock_model, mock_agent):
        mock_os.return_value.indices.get_mapping.return_value = {"idx": {}}
        mock_agent.return_value.side_effect = RuntimeError("bedrock unavailable")

        result = AgenticSearchAgent("https://localhost:9200")(
            "q", context={"index_name": "idx"}
        )
        assert result == FALLBACK_DSL
        assert json.loads(result) == MATCH_ALL

    @patch("agents.agentic_search.agent.create_model")
    def test_missing_index_name_falls_back_before_fetch(self, mock_model):
        with patch("agents.agentic_search.agent.OpenSearch") as mock_os:
            result = AgenticSearchAgent("https://localhost:9200")("q", context={})
            assert result == FALLBACK_DSL
            mock_os.assert_not_called()  # guarded before touching the cluster

    @patch("agents.agentic_search.agent.create_model")
    def test_blank_query_falls_back_before_fetch(self, mock_model):
        with patch("agents.agentic_search.agent.OpenSearch") as mock_os:
            result = AgenticSearchAgent("https://localhost:9200")(
                "   ", context={"index_name": "idx"}
            )
            assert result == FALLBACK_DSL
            mock_os.assert_not_called()

    @patch("agents.agentic_search.agent.create_model")
    def test_non_dict_context_falls_back(self, mock_model):
        result = AgenticSearchAgent("https://localhost:9200")("q", context=["not", "a", "dict"])
        assert result == FALLBACK_DSL

    @patch("agents.agentic_search.agent.create_model")
    def test_unknown_strategy_falls_back_before_fetch(self, mock_model):
        with patch("agents.agentic_search.agent.OpenSearch") as mock_os:
            result = AgenticSearchAgent("https://localhost:9200")(
                "q", context={"index_name": "idx", "strategy": "does_not_exist"}
            )
            assert result == FALLBACK_DSL
            mock_os.assert_not_called()

    @patch("agents.agentic_search.agent.create_model")
    def test_non_object_strategy_output_falls_back(self, mock_model, monkeypatch):
        # A strategy that returns a non-object (e.g. a list) is not usable DSL.
        stub = MagicMock()
        stub.generate.return_value = ["not", "an", "object"]
        monkeypatch.setitem(STRATEGIES, "stub", stub)
        with patch("agents.agentic_search.agent.OpenSearch") as mock_os:
            mock_os.return_value.indices.get_mapping.return_value = {"idx": {}}
            result = AgenticSearchAgent("https://localhost:9200")(
                "q", context={"index_name": "idx", "strategy": "stub"}
            )
            assert result == FALLBACK_DSL


class TestContract:
    """The agent's contract with the orchestrator and the caching design."""

    def test_declares_context_awareness(self):
        # The orchestrator keys context/auth forwarding off this flag.
        assert AgenticSearchAgent.accepts_invoke_context is True

    def test_system_prompt_ends_with_cache_point(self):
        # The prompt-caching win depends on this trailing block surviving.
        assert SYSTEM_BLOCKS[-1] == {"cachePoint": {"type": "default"}}
        assert "text" in SYSTEM_BLOCKS[0]

    @patch("agents.agentic_search.strategies.direct_dsl.Agent")
    @patch("agents.agentic_search.agent.create_model")
    @patch("agents.agentic_search.agent.OpenSearch")
    def test_uses_structured_output_model_and_cached_blocks(
        self, mock_os, mock_model, mock_agent
    ):
        # Guards the two details a refactor could silently drop: the cache-point
        # blocks reach the Agent, and generation uses structured_output_model
        # (not the deprecated structured_output(), which drops the cache point).
        mock_os.return_value.indices.get_mapping.return_value = {"idx": {}}
        mock_agent.return_value.return_value.structured_output = _emit(MATCH_ALL)

        AgenticSearchAgent("https://localhost:9200")("q", context={"index_name": "idx"})

        assert mock_agent.call_args.kwargs["system_prompt"] is SYSTEM_BLOCKS
        assert mock_agent.return_value.call_args.kwargs["structured_output_model"] is EmitSearch
