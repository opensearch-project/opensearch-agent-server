"""Unit tests for request input validation — tool-result size bound (issue #138).

Verifies that ``ValidatedRunAgentInput`` caps client-returned tool results at the API
boundary so oversized continuation input can't bloat the model context window.
"""

from __future__ import annotations

import pytest
from ag_ui.core import ToolMessage, UserMessage
from pydantic import ValidationError

from server.validators import _MAX_TOOL_RESULT_BYTES, ValidatedRunAgentInput

pytestmark = pytest.mark.unit


def _input_with_messages(messages: list) -> ValidatedRunAgentInput:
    return ValidatedRunAgentInput(
        thread_id="t1",
        run_id="r1",
        messages=messages,
        state={},
        tools=[],
        context=[],
        forwarded_props={},
    )


class TestToolResultSizeBound:
    """The client-returned tool-result cap (issue #138)."""

    def test_rejects_oversized_tool_result(self) -> None:
        """A tool result over the cap is rejected at the API boundary."""
        huge = "x" * (_MAX_TOOL_RESULT_BYTES + 1)
        msg = ToolMessage(id="m1", role="tool", content=huge, toolCallId="tc1")
        with pytest.raises(ValidationError, match="tool result is too large"):
            _input_with_messages([msg])

    def test_accepts_tool_result_within_bound(self) -> None:
        """A normal-sized tool result passes."""
        msg = ToolMessage(id="m1", role="tool", content="ok", toolCallId="tc1")
        vi = _input_with_messages([msg])
        assert len(vi.messages) == 1

    def test_accepts_tool_result_at_bound(self) -> None:
        """A tool result exactly at the cap is allowed (boundary is inclusive)."""
        exact = "x" * _MAX_TOOL_RESULT_BYTES
        msg = ToolMessage(id="m1", role="tool", content=exact, toolCallId="tc1")
        vi = _input_with_messages([msg])
        assert len(vi.messages) == 1

    def test_non_tool_message_not_bounded(self) -> None:
        """The cap applies only to tool-role messages, not ordinary user content."""
        big_user = "x" * (_MAX_TOOL_RESULT_BYTES + 1)
        msg = UserMessage(id="m1", role="user", content=big_user)
        vi = _input_with_messages([msg])
        assert len(vi.messages) == 1
