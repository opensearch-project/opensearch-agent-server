# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for utils.activity_monitor.AGUIActivityMonitor."""

from __future__ import annotations

import pytest

from utils.activity_monitor import AGUIActivityMonitor

pytestmark = pytest.mark.unit


class TestActivityMonitor:
    def _monitor(self) -> AGUIActivityMonitor:
        return AGUIActivityMonitor(run_id="run-1", thread_id="thread-1")

    def test_init(self):
        m = self._monitor()
        assert m.run_id == "run-1"
        assert m.thread_id == "thread-1"
        assert m.tool_call_count == 0
        assert m.activity_log == []
        assert m.active_tool_calls == {}

    def test_track_start_without_arguments(self):
        m = self._monitor()
        m.track_tool_call_start("tc1", "SearchTool")
        assert m.tool_call_count == 1
        assert "tc1" in m.active_tool_calls
        assert m.active_tool_calls["tc1"]["tool"] == "SearchTool"

    def test_track_start_with_arguments(self):
        m = self._monitor()
        m.track_tool_call_start("tc1", "SearchTool", arguments={"q": "x"})
        assert m.active_tool_calls["tc1"]["arguments"] == {"q": "x"}

    def test_track_end_success(self):
        m = self._monitor()
        m.track_tool_call_start("tc1", "SearchTool")
        m.track_tool_call_end("tc1", success=True)
        assert "tc1" not in m.active_tool_calls
        assert m.activity_log[0]["status"] == "success"
        assert "error" not in m.activity_log[0]

    def test_track_end_failure_with_error(self):
        m = self._monitor()
        m.track_tool_call_start("tc1", "SearchTool")
        m.track_tool_call_end("tc1", success=False, error="boom")
        assert m.activity_log[0]["status"] == "error"
        assert m.activity_log[0]["error"] == "boom"

    def test_track_end_unknown_id_is_noop(self):
        m = self._monitor()
        m.track_tool_call_end("does-not-exist")
        assert m.activity_log == []

    def test_get_summary_empty(self):
        m = self._monitor()
        summary = m.get_summary()
        assert summary["total_calls"] == 0
        assert summary["successful"] == 0
        assert summary["failed"] == 0
        assert summary["avg_duration"] == 0.0

    def test_get_summary_populated(self):
        m = self._monitor()
        m.track_tool_call_start("tc1", "A")
        m.track_tool_call_end("tc1", success=True)
        m.track_tool_call_start("tc2", "B")
        m.track_tool_call_end("tc2", success=False, error="e")
        summary = m.get_summary()
        assert summary["total_calls"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert summary["total_duration"] >= 0.0
        assert summary["avg_duration"] >= 0.0

    def test_get_remaining_tool_calls(self):
        m = self._monitor()
        m.track_tool_call_start("tc1", "A")
        m.track_tool_call_start("tc2", "B")
        assert set(m.get_remaining_tool_calls()) == {"tc1", "tc2"}

    def test_complete_remaining_tool_calls(self):
        m = self._monitor()
        m.track_tool_call_start("tc1", "A")
        m.track_tool_call_start("tc2", "B")
        m.complete_remaining_tool_calls()
        assert m.get_remaining_tool_calls() == []
        assert all(e["status"] == "error" for e in m.activity_log)

    def test_log_summary_no_failures(self):
        m = self._monitor()
        m.track_tool_call_start("tc1", "A")
        m.track_tool_call_end("tc1", success=True)
        m.log_summary()  # should not raise

    def test_log_summary_with_failures(self):
        m = self._monitor()
        m.track_tool_call_start("tc1", "A")
        m.track_tool_call_end("tc1", success=False, error="oops")
        m.log_summary()  # exercises the failed-calls logging branch
