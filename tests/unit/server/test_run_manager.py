# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.run_manager (RunManager lifecycle)."""

from __future__ import annotations

import asyncio

import pytest

import server.run_manager as rm_mod
from server.run_manager import RunManager, get_run_manager

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the RunManager singleton before and after each test for isolation."""
    rm_mod.RunManager._instance = None
    rm_mod._run_manager = None
    yield
    rm_mod.RunManager._instance = None
    rm_mod._run_manager = None


async def _forever() -> None:
    await asyncio.Event().wait()


async def _running_task() -> asyncio.Task:
    task = asyncio.create_task(_forever())
    await asyncio.sleep(0)  # let the task start running
    return task


async def _done_task() -> asyncio.Task:
    async def _noop() -> None:
        return

    task = asyncio.create_task(_noop())
    await task
    return task


def test_get_run_manager_is_singleton():
    """get_run_manager returns the same instance on repeated calls."""
    a = get_run_manager()
    b = get_run_manager()
    assert a is b
    assert isinstance(a, RunManager)


def test_init_short_circuits_when_already_initialized():
    """Calling RunManager() again returns the initialized singleton without resetting state."""
    mgr = get_run_manager()
    mgr._active_runs["sentinel"] = object()  # type: ignore[assignment]
    again = RunManager()
    assert again is mgr
    assert "sentinel" in again._active_runs


@pytest.mark.asyncio
async def test_register_and_active_and_count():
    """A registered non-done run is active and counted."""
    mgr = get_run_manager()
    task = await _running_task()
    try:
        await mgr.register_run("r1", task)
        assert await mgr.is_run_active("r1") is True
        assert await mgr.get_active_run_count() == 1
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_unregister_removes_run():
    """Unregistering a run removes it from active tracking."""
    mgr = get_run_manager()
    task = await _running_task()
    try:
        await mgr.register_run("r1", task)
        await mgr.unregister_run("r1")
        assert await mgr.is_run_active("r1") is False
        assert await mgr.get_active_run_count() == 0
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_is_run_active_unknown_returns_false():
    """is_run_active for an unregistered run is False."""
    mgr = get_run_manager()
    assert await mgr.is_run_active("nope") is False


@pytest.mark.asyncio
async def test_cancel_nonexistent_run_returns_false():
    """Canceling a run that was never registered returns False."""
    mgr = get_run_manager()
    assert await mgr.cancel_run("ghost") is False


@pytest.mark.asyncio
async def test_cancel_active_run_then_already_canceled():
    """Canceling an active run succeeds; a second cancel reports already-canceled."""
    mgr = get_run_manager()
    task = await _running_task()
    try:
        await mgr.register_run("r1", task)
        assert await mgr.cancel_run("r1", reason="test") is True
        assert await mgr.is_run_canceled("r1") is True
        # Second attempt hits the already-canceled branch.
        assert await mgr.cancel_run("r1") is False
    finally:
        task.cancel()


@pytest.mark.asyncio
async def test_cancel_already_done_run_returns_false_and_unregisters():
    """Canceling a run whose task already finished returns False and unregisters it."""
    mgr = get_run_manager()
    task = await _done_task()
    await mgr.register_run("r1", task)
    assert await mgr.cancel_run("r1") is False
    # It should have been unregistered by the already-done branch.
    assert await mgr.is_run_active("r1") is False


@pytest.mark.asyncio
async def test_is_run_canceled_false_for_unknown():
    """is_run_canceled is False when the run was never canceled."""
    mgr = get_run_manager()
    assert await mgr.is_run_canceled("unknown") is False


@pytest.mark.asyncio
async def test_get_active_run_count_cleans_completed():
    """get_active_run_count prunes completed tasks."""
    mgr = get_run_manager()
    done = await _done_task()
    await mgr.register_run("done", done)
    assert await mgr.get_active_run_count() == 0
    assert await mgr.is_run_active("done") is False


@pytest.mark.asyncio
async def test_cleanup_completed_runs_returns_count():
    """cleanup_completed_runs removes finished runs and returns how many."""
    mgr = get_run_manager()
    done = await _done_task()
    running = await _running_task()
    try:
        await mgr.register_run("done", done)
        await mgr.register_run("running", running)
        cleaned = await mgr.cleanup_completed_runs()
        assert cleaned == 1
        assert await mgr.is_run_active("running") is True
    finally:
        running.cancel()
