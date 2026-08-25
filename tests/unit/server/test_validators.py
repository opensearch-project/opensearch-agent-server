# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Unit tests for request input validation — tool-result size bound (issue #138).

Verifies that ``ValidatedRunAgentInput`` caps client-returned tool results at the API
boundary so oversized continuation input can't bloat the model context window.
"""

from __future__ import annotations

from types import SimpleNamespace

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


# --- additional coverage tests (merged) ---
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.validators.ValidatedRunAgentInput.

Field/model validators are exercised through full model construction where
possible; the message-shape branches (missing role/content, non-serializable
tool result) are exercised by invoking the raw field-validator function with
lightweight stand-in message objects.
"""


# Raw (unbound) message field-validator function for direct branch testing.
_VALIDATE_MESSAGES = ValidatedRunAgentInput.__pydantic_decorators__.field_validators[
    "validate_messages"
].func


def _valid(**overrides) -> dict:
    base = dict(
        thread_id="t1",
        run_id="r1",
        messages=[UserMessage(id="m1", role="user", content="hi")],
        state={},
        tools=[],
        context=[],
        forwarded_props={},
    )
    base.update(overrides)
    return base


class TestFieldValidators:
    def test_valid_input(self):
        vi = ValidatedRunAgentInput(**_valid())
        assert vi.thread_id == "t1"
        assert vi.run_id == "r1"

    def test_thread_id_empty_rejected(self):
        with pytest.raises(ValidationError, match="thread_id must be a non-empty"):
            ValidatedRunAgentInput(**_valid(thread_id="   "))

    def test_run_id_empty_rejected(self):
        with pytest.raises(ValidationError, match="run_id must be a non-empty"):
            ValidatedRunAgentInput(**_valid(run_id="  "))

    def test_thread_id_stripped(self):
        vi = ValidatedRunAgentInput(**_valid(thread_id="  t1  "))
        assert vi.thread_id == "t1"

    def test_parent_run_id_valid(self):
        vi = ValidatedRunAgentInput(**_valid(parent_run_id="  parent  "))
        assert vi.parent_run_id == "parent"

    def test_parent_run_id_none(self):
        vi = ValidatedRunAgentInput(**_valid(parent_run_id=None))
        assert vi.parent_run_id is None

    def test_parent_run_id_empty_rejected(self):
        with pytest.raises(ValidationError, match="parent_run_id must be a non-empty"):
            ValidatedRunAgentInput(**_valid(parent_run_id="   "))

    def test_messages_empty_rejected(self):
        with pytest.raises(ValidationError):
            ValidatedRunAgentInput(**_valid(messages=[]))

    def test_thread_and_run_id_same_rejected(self):
        with pytest.raises(ValidationError, match="must be different"):
            ValidatedRunAgentInput(**_valid(thread_id="same", run_id="same"))


class TestToolResultBound:
    def test_oversized_tool_result_rejected(self):
        huge = "x" * (_MAX_TOOL_RESULT_BYTES + 1)
        msg = ToolMessage(id="m1", role="tool", content=huge, toolCallId="tc1")
        with pytest.raises(ValidationError, match="tool result is too large"):
            ValidatedRunAgentInput(**_valid(messages=[msg]))

    def test_tool_result_at_bound_ok(self):
        exact = "x" * _MAX_TOOL_RESULT_BYTES
        msg = ToolMessage(id="m1", role="tool", content=exact, toolCallId="tc1")
        vi = ValidatedRunAgentInput(**_valid(messages=[msg]))
        assert len(vi.messages) == 1


class TestValidateMessagesRawBranches:
    """Directly exercise message-shape branches with stand-in messages."""

    def test_missing_role_rejected(self):
        msg = SimpleNamespace(role="")
        with pytest.raises(Exception, match="role is required"):
            _VALIDATE_MESSAGES([msg], None)

    def test_empty_list_rejected(self):
        # Field(min_length=1) fires before the validator during full-model
        # construction, so exercise the validator's own empty-guard directly.
        with pytest.raises(Exception, match="messages must be a non-empty"):
            _VALIDATE_MESSAGES([], None)

    def test_missing_content_rejected(self):
        msg = SimpleNamespace(role="user", content=None, tool_calls=None)
        with pytest.raises(Exception, match="content is required"):
            _VALIDATE_MESSAGES([msg], None)

    def test_assistant_with_tool_calls_allows_missing_content(self):
        msg = SimpleNamespace(role="assistant", content=None, tool_calls=[{"id": "c1"}])
        result = _VALIDATE_MESSAGES([msg], None)
        assert result == [msg]

    def test_tool_result_dict_content_too_large(self):
        big = "x" * (_MAX_TOOL_RESULT_BYTES + 10)
        msg = SimpleNamespace(role="tool", content={"data": big}, tool_calls=None)
        with pytest.raises(Exception, match="tool result is too large"):
            _VALIDATE_MESSAGES([msg], None)

    def test_tool_result_unserializable_content_too_large(self):
        # Non-str keys make json.dumps raise TypeError even with default=str,
        # which the validator treats as over-limit.
        msg = SimpleNamespace(role="tool", content={(1, 2): "v"}, tool_calls=None)
        with pytest.raises(Exception, match="tool result is too large"):
            _VALIDATE_MESSAGES([msg], None)


class TestConversion:
    def test_to_run_agent_input(self):
        vi = ValidatedRunAgentInput(**_valid(parent_run_id="p1"))
        rai = vi.to_run_agent_input()
        assert rai.thread_id == "t1"
        assert rai.run_id == "r1"
        assert rai.parent_run_id == "p1"

    def test_from_run_agent_input(self):
        vi = ValidatedRunAgentInput(**_valid())
        rai = vi.to_run_agent_input()
        vi2 = ValidatedRunAgentInput.from_run_agent_input(rai)
        assert vi2.thread_id == "t1"
        assert vi2.run_id == "r1"
