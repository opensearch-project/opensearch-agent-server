# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Run Manager for tracking and canceling active AG-UI runs (run_id -> asyncio.Task).

Manages active runs by tracking run_id -> asyncio.Task mappings,
allowing runs to be canceled on demand.

**Key Features:**
- Singleton pattern for global access
- Thread-safe operations (asyncio.Lock)
- Run cancellation support
- Active run tracking
- Automatic cleanup of completed runs

**Usage Example:**
```python
from server.run_manager import get_run_manager

run_manager = get_run_manager()

# Register a run
task = asyncio.create_task(my_event_generator())
await run_manager.register_run("run-123", task)

# Check if run is active
is_active = await run_manager.is_run_active("run-123")

# Cancel a run
canceled = await run_manager.cancel_run("run-123", reason="User requested")

# Unregister when done
await run_manager.unregister_run("run-123")
```
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from utils.logging_helpers import get_logger, log_info_event, log_warning_event

logger = get_logger(__name__)


class RunManager:
    """Manages active runs and provides cancellation functionality.

    Thread-safe singleton pattern for tracking active runs across the application.
    Each run is tracked as an asyncio.Task that can be canceled.
    """

    _instance: RunManager | None = None
    _lock = asyncio.Lock()

    def __new__(cls) -> RunManager:
        """Singleton pattern - ensures only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the run manager."""
        if self._initialized:
            return

        # Dictionary mapping run_id -> asyncio.Task
        self._active_runs: dict[str, asyncio.Task] = {}
        # Dictionary mapping run_id -> cancellation timestamp
        self._canceled_runs: dict[str, datetime] = {}
        self._initialized = True

    async def register_run(self, run_id: str, task: asyncio.Task) -> None:
        """Register an active run.

        Args:
            run_id: Run identifier
            task: asyncio.Task wrapping the event stream
        """
        async with self._lock:
            self._active_runs[run_id] = task
            log_info_event(
                logger,
                "Registered active run.",
                "ag_ui.run_registered",
                run_id=run_id,
            )

    def _unregister_run_internal(self, run_id: str) -> None:
        """Internal method to unregister a run without acquiring lock.

        Caller must hold self._lock before calling this method.

        Args:
            run_id: Run identifier
        """
        if run_id in self._active_runs:
            del self._active_runs[run_id]
            log_info_event(
                logger,
                "Unregistered run.",
                "ag_ui.run_unregistered",
                run_id=run_id,
            )
        # Clean up canceled runs after a delay
        if run_id in self._canceled_runs:
            del self._canceled_runs[run_id]

    async def unregister_run(self, run_id: str) -> None:
        """Unregister a run when it completes.

        Args:
            run_id: Run identifier
        """
        async with self._lock:
            self._unregister_run_internal(run_id)

    async def cancel_run(
        self, run_id: str, reason: str = "User requested cancellation"
    ) -> bool:
        """Cancel an active run.

        Args:
            run_id: Run identifier
            reason: Reason for cancellation (for logging)

        Returns:
            True if run was found and canceled, False otherwise
        """
        async with self._lock:
            if run_id not in self._active_runs:
                log_warning_event(
                    logger,
                    "Attempted to cancel non-existent run.",
                    "ag_ui.run_cancel_not_found",
                    run_id=run_id,
                )
                return False

            # Check if already canceled
            if run_id in self._canceled_runs:
                log_warning_event(
                    logger,
                    "Attempted to cancel already canceled run.",
                    "ag_ui.run_cancel_already_canceled",
                    run_id=run_id,
                )
                return False

            task = self._active_runs[run_id]

            # Check if task is already done
            if task.done():
                log_warning_event(
                    logger,
                    "Attempted to cancel already completed run.",
                    "ag_ui.run_cancel_already_done",
                    run_id=run_id,
                )
                # Call internal method directly since we already hold the lock
                self._unregister_run_internal(run_id)
                return False

            # Mark as canceled
            self._canceled_runs[run_id] = datetime.now()

            # Cancel the task
            canceled = task.cancel()

            if canceled:
                log_info_event(
                    logger,
                    "Canceled run.",
                    "ag_ui.run_canceled",
                    run_id=run_id,
                    reason=reason,
                )
            else:
                log_warning_event(
                    logger,
                    "Failed to cancel run (may have completed).",
                    "ag_ui.run_cancel_failed",
                    run_id=run_id,
                )

            return canceled

    async def is_run_active(self, run_id: str) -> bool:
        """Check if a run is currently active.

        Args:
            run_id: Run identifier

        Returns:
            True if run is active, False otherwise
        """
        async with self._lock:
            if run_id not in self._active_runs:
                return False

            task = self._active_runs[run_id]
            return not task.done()

    async def is_run_canceled(self, run_id: str) -> bool:
        """Check if a run was canceled.

        Args:
            run_id: Run identifier

        Returns:
            True if run was canceled, False otherwise
        """
        async with self._lock:
            return run_id in self._canceled_runs

    async def get_active_run_count(self) -> int:
        """Get the number of currently active runs.

        Returns:
            Number of active runs
        """
        async with self._lock:
            # Clean up completed tasks
            completed_runs = [
                run_id for run_id, task in self._active_runs.items() if task.done()
            ]
            for run_id in completed_runs:
                # Call internal method directly since we already hold the lock
                self._unregister_run_internal(run_id)

            return len(self._active_runs)

    async def cleanup_completed_runs(self) -> int:
        """Clean up completed runs from tracking.

        Returns:
            Number of runs cleaned up
        """
        async with self._lock:
            completed_runs = [
                run_id for run_id, task in self._active_runs.items() if task.done()
            ]
            for run_id in completed_runs:
                # Call internal method directly since we already hold the lock
                self._unregister_run_internal(run_id)

            return len(completed_runs)


# Global singleton instance
_run_manager: RunManager | None = None


def get_run_manager() -> RunManager:
    """Get the global RunManager singleton instance.

    Prefer this over constructing RunManager() directly.

    Returns:
        RunManager instance
    """
    global _run_manager
    if _run_manager is None:
        _run_manager = RunManager()
    return _run_manager
