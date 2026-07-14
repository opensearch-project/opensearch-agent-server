"""Unit tests for the self-contained BedrockDslGenerator.

Mocks boto3, opensearch-py, and the strands Agent so the generator's wiring is
verified with no cluster, no AWS, and no LLM:
1. It reads the index mapping and feeds it into the prompt.
2. It forces an EmitSearch structured-output call and returns dsl as a JSON string.
3. A bearer token is passed to the OpenSearch client when provided.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def _make_generator():
    # Import inside the test so module-level heavy imports can be patched first.
    from server.dsl_generator import BedrockDslGenerator, EmitSearch

    return BedrockDslGenerator, EmitSearch


@patch("server.dsl_generator.Agent")
@patch("server.dsl_generator.create_model")
@patch("server.dsl_generator.OpenSearch")
def test_generate_reads_mapping_and_returns_dsl_string(mock_os, mock_model, mock_agent):
    BedrockDslGenerator, EmitSearch = _make_generator()

    mock_os.return_value.indices.get_mapping.return_value = {
        "idx": {"mappings": {"properties": {"status": {"type": "keyword"}}}}
    }
    dsl_obj = {"size": 10, "query": {"term": {"status": "active"}}}
    mock_agent.return_value.return_value.structured_output = EmitSearch(
        reason="status=active -> term", dsl=dsl_obj
    )

    gen = BedrockDslGenerator("https://localhost:9200")
    out = gen.generate("active items", "idx", auth_token="svc-token")

    # Returns the dsl serialized to a STRING (what the passthrough envelope needs).
    assert isinstance(out, str)
    assert json.loads(out) == dsl_obj

    # The mapping was read for the right index and fed into the prompt.
    mock_os.return_value.indices.get_mapping.assert_called_once_with(index="idx")
    prompt_arg = mock_agent.return_value.call_args[0][0]
    assert "active items" in prompt_arg and "status" in prompt_arg


@patch("server.dsl_generator.Agent")
@patch("server.dsl_generator.create_model")
@patch("server.dsl_generator.OpenSearch")
def test_bearer_token_forwarded_to_client(mock_os, mock_model, mock_agent):
    BedrockDslGenerator, EmitSearch = _make_generator()
    mock_os.return_value.indices.get_mapping.return_value = {"idx": {}}
    mock_agent.return_value.return_value.structured_output = EmitSearch(
        reason="r", dsl={"query": {"match_all": {}}}
    )

    BedrockDslGenerator("https://localhost:9200").generate("q", "idx", auth_token="tok")

    _, kwargs = mock_os.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer tok"}


@patch("server.dsl_generator.Agent")
@patch("server.dsl_generator.create_model")
@patch("server.dsl_generator.OpenSearch")
def test_no_token_means_no_auth_header(mock_os, mock_model, mock_agent):
    BedrockDslGenerator, EmitSearch = _make_generator()
    mock_os.return_value.indices.get_mapping.return_value = {"idx": {}}
    mock_agent.return_value.return_value.structured_output = EmitSearch(
        reason="r", dsl={"query": {"match_all": {}}}
    )

    BedrockDslGenerator("https://localhost:9200").generate("q", "idx")

    _, kwargs = mock_os.call_args
    assert kwargs["headers"] is None
