# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
AG-UI Activity Monitor

Tracks tool calls and agent activity for AG-UI server.
Provides activity summaries similar to Chainlit's AgentActivityMonitor
but without Chainlit dependencies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from utils.logging_helpers import (
    get_logger,
    log_debug_event,
    log_info_event,
    log_warning_event,
)

logger = get_logger(__name__)


class AGUIActivityMonitor:
    """
    Monitors agent activity for AG-UI server.

    Tracks tool calls, success/failure rates, and durations
    to provide activity summaries for debugging and metrics.
    """

    def __init__(self, run_id: str, thread_id: str) -> None:
        """
        Initialize activity monitor for a run.

        Args:
            run_id: Run identifier
            thread_id: Thread identifier
        """
        self.run_id = run_id
        self.thread_id = thread_id
        self.tool_call_count = 0
        self.activity_log: list[dict[str, Any]] = []
        self.active_tool_calls: dict[str, dict[str, Any]] = {}

    def track_tool_call_start(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any | None] = None,
    ) -> None:
        """
        Track the start of a tool call.

        Args:
            tool_call_id: Unique identifier for the tool call
            tool_name: Name of the tool being called
            arguments: Tool arguments (optional)
        """
        self.tool_call_count += 1
        call_num = self.tool_call_count

        self.active_tool_calls[tool_call_id] = {
            "call_num": call_num,
            "tool": tool_name,
            "start_time": datetime.now(),
            "arguments": arguments,
        }

        log_debug_event(
            logger,
            "Tool call started.",
            "activity_monitor.tool_call_started",
            tool_name=tool_name,
            call_num=call_num,
            tool_call_id=tool_call_id,
            run_id=self.run_id,
        )

    def track_tool_call_end(
        self, tool_call_id: str, success: bool = True, error: str | None = None
    ) -> None:
        """
        Track the end of a tool call.

        Args:
            tool_call_id: Unique identifier for the tool call
            success: Whether the tool call succeeded
            error: Error message if failed (optional)
        """
        if tool_call_id not in self.active_tool_calls:
            log_warning_event(
                logger,
                "Tool call end tracked for unknown tool_call_id.",
                "activity_monitor.tool_call_end_unknown_id",
                tool_call_id=tool_call_id,
            )
            return

        call_info = self.active_tool_calls.pop(tool_call_id)
        start_time = call_info["start_time"]
        duration = (datetime.now() - start_time).total_seconds()
        tool_name = call_info["tool"]
        call_num = call_info["call_num"]

        # Log activity
        activity_entry = {
            "call_num": call_num,
            "tool": tool_name,
            "status": "success" if success else "error",
            "duration": duration,
        }

        if error:
            activity_entry["error"] = error

        self.activity_log.append(activity_entry)

        status_icon = "✓" if success else "❌"
        log_debug_event(
            logger,
            "Tool call ended.",
            "activity_monitor.tool_call_ended",
            tool_name=tool_name,
            call_num=call_num,
            tool_call_id=tool_call_id,
            status_icon=status_icon,
            duration=duration,
            run_id=self.run_id,
        )

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of all activity.

        Returns:
            Dictionary with activity summary statistics (run_id, thread_id,
            total_calls, successful, failed, total_duration, avg_duration).
            Note: total_calls is the number of tool call starts; successful +
            failed may be less if some calls are still active.
        """
        successful = sum(1 for log in self.activity_log if log["status"] == "success")
        failed = sum(1 for log in self.activity_log if log["status"] == "error")
        total_duration = sum(log["duration"] for log in self.activity_log)

        # Calculate average duration
        avg_duration = (
            total_duration / len(self.activity_log) if self.activity_log else 0.0
        )

        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "total_calls": self.tool_call_count,
            "successful": successful,
            "failed": failed,
            "total_duration": total_duration,
            "avg_duration": avg_duration,
        }

    def get_remaining_tool_calls(self) -> list[str]:
        """
        Get list of tool call IDs that are still active.

        Returns:
            List of tool_call_id strings for active tool calls
        """
        return list(self.active_tool_calls.keys())

    def complete_remaining_tool_calls(
        self, error: str = "Run completed before tool call finished"
    ) -> None:
        """
        Complete all remaining active tool calls as failed.

        Args:
            error: Error message to use for incomplete tool calls
        """
        for tool_call_id in list(self.active_tool_calls.keys()):
            self.track_tool_call_end(
                tool_call_id=tool_call_id, success=False, error=error
            )

    def log_summary(self) -> None:
        """
        Log the activity summary.

        Uses structured logging format similar to Chainlit's activity summary.
        """
        summary = self.get_summary()

        log_info_event(
            logger,
            "Activity summary.",
            "activity_monitor.summary",
            run_id=summary["run_id"],
            thread_id=summary["thread_id"],
            total_calls=summary["total_calls"],
            successful=summary["successful"],
            failed=summary["failed"],
            total_duration=summary["total_duration"],
            avg_duration=summary["avg_duration"],
        )

        # Log individual tool calls if any failed
        if summary["failed"] > 0:
            failed_calls = [
                log for log in self.activity_log if log["status"] == "error"
            ]
            for call in failed_calls:
                log_warning_event(
                    logger,
                    "Failed tool call.",
                    "activity_monitor.tool_call_failed",
                    call_num=call["call_num"],
                    tool=call["tool"],
                    duration=call["duration"],
                    error=call.get("error", "unknown"),
                )
