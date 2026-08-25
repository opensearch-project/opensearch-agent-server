# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for agents.agentic_search.strategies.forced_tool.

The Bedrock ``converse_stream`` client is fully mocked; no AWS calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agents.agentic_search.strategies.forced_tool import (
    forced_tool_fill,
    supports_forced_tool,
)

pytestmark = pytest.mark.unit


class _Result(BaseModel):
    answer: str


def _stream_events(chunks: list[str]):
    return [{"contentBlockDelta": {"delta": {"toolUse": {"input": c}}}} for c in chunks]


def _make_model(*, config: dict, stream_events):
    client = MagicMock()
    client.converse_stream.return_value = {"stream": stream_events}
    return SimpleNamespace(client=client, config=config)


class TestSupportsForcedTool:
    def test_true_for_bedrock_like_client(self):
        model = SimpleNamespace(
            client=SimpleNamespace(converse_stream=lambda **k: None)
        )
        assert supports_forced_tool(model) is True

    def test_false_when_client_none(self):
        assert supports_forced_tool(SimpleNamespace(client=None)) is False

    def test_false_when_no_converse_stream(self):
        model = SimpleNamespace(client=SimpleNamespace())
        assert supports_forced_tool(model) is False


class TestForcedToolFill:
    def test_success_with_precomputed_spec_and_temperature(self):
        spec = {
            "name": "Result",
            "description": "Emit the result.",
            "inputSchema": {"json": {"type": "object"}},
        }
        model = _make_model(
            config={"model_id": "m1", "temperature": 0.2},
            stream_events=_stream_events(['{"answer":', ' "hi"}']),
        )
        result = forced_tool_fill(
            model=model,
            schema_model=_Result,
            system_blocks=[{"text": "sys"}],
            user_message="hello",
            tool_spec=spec,
        )
        assert isinstance(result, _Result)
        assert result.answer == "hi"
        # Temperature applied to inferenceConfig.
        kwargs = model.client.converse_stream.call_args[1]
        assert kwargs["inferenceConfig"] == {"temperature": 0.2}
        assert kwargs["toolConfig"]["toolChoice"] == {"tool": {"name": "Result"}}

    def test_success_computes_spec_when_not_given(self):
        model = _make_model(
            config={"model_id": "m1"},  # no temperature
            stream_events=_stream_events(['{"answer": "auto"}']),
        )
        result = forced_tool_fill(
            model=model,
            schema_model=_Result,
            system_blocks=[{"text": "sys"}],
            user_message="hello",
        )
        assert result.answer == "auto"
        # No temperature -> no inferenceConfig key.
        assert "inferenceConfig" not in model.client.converse_stream.call_args[1]

    def test_empty_input_raises(self):
        model = _make_model(
            config={"model_id": "m1"},
            stream_events=_stream_events(["   "]),
        )
        with pytest.raises(ValueError, match="produced no input"):
            forced_tool_fill(
                model=model,
                schema_model=_Result,
                system_blocks=[],
                user_message="x",
            )

    def test_invalid_input_raises_validation_error(self):
        model = _make_model(
            config={"model_id": "m1"},
            stream_events=_stream_events(['{"wrong": 1}']),
        )
        with pytest.raises(ValueError, match="failed validation"):
            forced_tool_fill(
                model=model,
                schema_model=_Result,
                system_blocks=[],
                user_message="x",
            )
