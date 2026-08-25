# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage-focused unit tests for utils.persistence.AGUIPersistence.

These tests exercise the SQLAlchemy-backed persistence layer end to end using
an in-memory SQLite database (``sqlite:///:memory:``) so real engine/session,
table-creation, CRUD, serialization and cascade-delete code paths run without
touching disk or any external service. Error/rollback branches are covered by
swapping ``_get_session`` for a mock whose ``commit``/``query`` raises.

All tests are synchronous (the module under test uses the *sync* SQLAlchemy
engine); ``asyncio_mode=auto`` is irrelevant here.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from utils.persistence import AGUIPersistence, Event, Message, Run, Thread

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> AGUIPersistence:
    """A fresh in-memory persistence instance with tables created.

    Using ``:memory:`` keeps SQLAlchemy's SingletonThreadPool so every session
    created within this (single) test thread shares the same connection and
    therefore the same schema/data.
    """
    return AGUIPersistence(db_path=":memory:")


@pytest.fixture
def failing_session_store(store: AGUIPersistence):
    """Return (store, mock_session) where the store hands out a mock session
    whose ``commit`` raises, so save_* rollback/raise branches are exercised."""
    fake = MagicMock()
    fake.commit.side_effect = RuntimeError("boom")
    # query(...).filter(...).first() returns a truthy mock so "found" branches run
    store._get_session = lambda: fake  # type: ignore[method-assign]
    return store, fake


# ---------------------------------------------------------------------------
# Construction / engine + directory setup
# ---------------------------------------------------------------------------


def test_init_memory_creates_tables(store: AGUIPersistence) -> None:
    assert store.db_path == ":memory:"
    assert store.engine is not None
    assert store.SessionLocal is not None
    # Schema exists: querying an empty table works and returns []
    assert store.get_threads() == []


def test_init_defaults_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # db_path=None -> read AG_UI_DB_PATH, and a non-empty dirname triggers makedirs
    db_file = tmp_path / "nested" / "dir" / "chat_history.db"
    monkeypatch.setenv("AG_UI_DB_PATH", str(db_file))
    p = AGUIPersistence(db_path=None)
    assert p.db_path == str(db_file)
    # _ensure_db_directory created the parent directory
    assert db_file.parent.is_dir()
    # And the sqlite file is usable
    p.save_thread("t-env")
    assert p.get_thread("t-env") is not None


def test_ensure_db_directory_no_dirname(monkeypatch: pytest.MonkeyPatch) -> None:
    # A bare filename has empty dirname -> makedirs branch skipped, no crash
    monkeypatch.delenv("AG_UI_DB_PATH", raising=False)
    p = AGUIPersistence(db_path=":memory:")
    assert p.db_path == ":memory:"


# ---------------------------------------------------------------------------
# Thread save / get / list
# ---------------------------------------------------------------------------


def test_save_thread_insert_and_get(store: AGUIPersistence) -> None:
    store.save_thread("t1", user_id="alice", metadata={"k": "v"})
    row = store.get_thread("t1")
    assert row is not None
    assert row["id"] == "t1"
    assert row["user_id"] == "alice"
    assert row["metadata"] == {"k": "v"}
    assert row["created_at"] is not None
    assert row["updated_at"] is not None


def test_save_thread_insert_without_metadata(store: AGUIPersistence) -> None:
    store.save_thread("t-nometa")
    row = store.get_thread("t-nometa")
    assert row is not None
    assert row["user_id"] is None
    assert row["metadata"] == {}


def test_save_thread_update_applies_provided_fields(store: AGUIPersistence) -> None:
    store.save_thread("t2", user_id="bob", metadata={"a": 1})
    store.save_thread("t2", user_id="carol", metadata={"a": 2, "b": 3})
    row = store.get_thread("t2")
    assert row["user_id"] == "carol"
    assert row["metadata"] == {"a": 2, "b": 3}


def test_save_thread_update_leaves_unchanged_when_none(store: AGUIPersistence) -> None:
    store.save_thread("t3", user_id="dave", metadata={"x": 9})
    # Update with all-None optionals: existing values must be preserved
    store.save_thread("t3", user_id=None, metadata=None)
    row = store.get_thread("t3")
    assert row["user_id"] == "dave"
    assert row["metadata"] == {"x": 9}


def test_get_thread_not_found(store: AGUIPersistence) -> None:
    assert store.get_thread("missing") is None


def test_get_threads_filter_and_pagination(store: AGUIPersistence) -> None:
    for i in range(5):
        store.save_thread(f"u1-{i}", user_id="u1")
    for i in range(3):
        store.save_thread(f"u2-{i}", user_id="u2")

    all_threads = store.get_threads()
    assert len(all_threads) == 8

    only_u1 = store.get_threads(user_id="u1")
    assert len(only_u1) == 5
    assert all(t["user_id"] == "u1" for t in only_u1)

    page = store.get_threads(limit=2, offset=1)
    assert len(page) == 2


def test_save_thread_rollback_on_error(failing_session_store) -> None:
    store, fake = failing_session_store
    with pytest.raises(RuntimeError):
        store.save_thread("t-err", user_id="x", metadata={"m": 1})
    fake.rollback.assert_called_once()
    fake.close.assert_called_once()


# ---------------------------------------------------------------------------
# Run save_start / save_finish / get
# ---------------------------------------------------------------------------


def test_save_run_start_and_get(store: AGUIPersistence) -> None:
    store.save_thread("th", user_id="u")
    store.save_run_start("r1", "th", metadata={"src": "test"})
    run = store.get_run("r1")
    assert run is not None
    assert run["id"] == "r1"
    assert run["thread_id"] == "th"
    assert run["status"] == "running"
    assert run["finished_at"] is None
    assert run["metadata"] == {"src": "test"}


def test_save_run_start_without_metadata(store: AGUIPersistence) -> None:
    store.save_thread("th", user_id="u")
    store.save_run_start("r-nometa", "th")
    run = store.get_run("r-nometa")
    assert run["metadata"] == {}


def test_save_run_finish_completed(store: AGUIPersistence) -> None:
    store.save_thread("th", user_id="u")
    store.save_run_start("r2", "th")
    store.save_run_finish("r2", status="completed")
    run = store.get_run("r2")
    assert run["status"] == "completed"
    assert run["finished_at"] is not None
    assert run["error_message"] is None


def test_save_run_finish_error_with_message(store: AGUIPersistence) -> None:
    store.save_thread("th", user_id="u")
    store.save_run_start("r3", "th")
    store.save_run_finish("r3", status="error", error_message="kaboom")
    run = store.get_run("r3")
    assert run["status"] == "error"
    assert run["error_message"] == "kaboom"


def test_save_run_finish_missing_run_logs_warning(store: AGUIPersistence) -> None:
    # No exception; hits the "run not found" warning branch
    store.save_run_finish("does-not-exist", status="completed")
    assert store.get_run("does-not-exist") is None


def test_get_run_not_found(store: AGUIPersistence) -> None:
    assert store.get_run("nope") is None


def test_get_runs_for_thread(store: AGUIPersistence) -> None:
    store.save_thread("th", user_id="u")
    for i in range(4):
        store.save_run_start(f"run-{i}", "th")
    runs = store.get_runs("th")
    assert len(runs) == 4
    assert {r["id"] for r in runs} == {f"run-{i}" for i in range(4)}
    # pagination
    assert len(store.get_runs("th", limit=2, offset=2)) == 2


def test_get_run_with_ownership_check(store: AGUIPersistence) -> None:
    store.save_thread("owned", user_id="owner")
    store.save_run_start("r-own", "owned")
    ok = store.get_run_with_ownership_check("r-own", "owner")
    assert ok is not None
    assert ok["id"] == "r-own"
    # wrong user -> None
    assert store.get_run_with_ownership_check("r-own", "intruder") is None
    # unknown run -> None
    assert store.get_run_with_ownership_check("ghost", "owner") is None


def test_save_run_start_rollback_on_error(failing_session_store) -> None:
    store, fake = failing_session_store
    with pytest.raises(RuntimeError):
        store.save_run_start("r-err", "th")
    fake.rollback.assert_called_once()
    fake.close.assert_called_once()


def test_save_run_finish_rollback_on_error(failing_session_store) -> None:
    store, fake = failing_session_store
    # query(...).first() returns a truthy mock -> enters found branch, commit raises
    with pytest.raises(RuntimeError):
        store.save_run_finish("r-err", status="completed")
    fake.rollback.assert_called_once()
    fake.close.assert_called_once()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def test_save_and_get_messages(store: AGUIPersistence) -> None:
    store.save_thread("th", user_id="u")
    store.save_run_start("run", "th")
    store.save_message("m1", "th", role="user", content="hello", run_id="run")
    store.save_message(
        "m2", "th", role="assistant", content="hi", run_id="run", metadata={"tok": 5}
    )
    store.save_message("m3", "th", role="user", content="orphan-run")

    msgs = store.get_messages("th")
    assert len(msgs) == 3
    by_id = {m["id"]: m for m in msgs}
    assert by_id["m1"]["role"] == "user"
    assert by_id["m1"]["content"] == "hello"
    assert by_id["m1"]["metadata"] == {}
    assert by_id["m2"]["metadata"] == {"tok": 5}
    assert by_id["m3"]["run_id"] is None


def test_get_messages_filtered_by_run(store: AGUIPersistence) -> None:
    store.save_thread("th", user_id="u")
    store.save_run_start("runA", "th")
    store.save_run_start("runB", "th")
    store.save_message("mA", "th", role="user", content="a", run_id="runA")
    store.save_message("mB", "th", role="user", content="b", run_id="runB")
    only_a = store.get_messages("th", run_id="runA")
    assert len(only_a) == 1
    assert only_a[0]["id"] == "mA"
    # pagination on unfiltered
    assert len(store.get_messages("th", limit=1, offset=1)) == 1


def test_save_message_rollback_on_error(failing_session_store) -> None:
    store, fake = failing_session_store
    with pytest.raises(RuntimeError):
        store.save_message("m-err", "th", role="user", content="x")
    fake.rollback.assert_called_once()
    fake.close.assert_called_once()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_save_and_get_events(store: AGUIPersistence) -> None:
    store.save_thread("th", user_id="u")
    store.save_run_start("run", "th")
    store.save_event("e1", "run", "TEXT_MESSAGE_START", {"foo": "bar"})
    store.save_event("e2", "run", "TOOL_CALL_START", {"tool": "search"})

    events = store.get_events("run")
    assert len(events) == 2
    by_id = {e["id"]: e for e in events}
    assert by_id["e1"]["event_type"] == "TEXT_MESSAGE_START"
    assert by_id["e1"]["event_data"] == {"foo": "bar"}
    assert by_id["e1"]["created_at"] is not None


def test_get_events_filtered_by_type(store: AGUIPersistence) -> None:
    store.save_thread("th", user_id="u")
    store.save_run_start("run", "th")
    store.save_event("e1", "run", "TYPE_A", {"n": 1})
    store.save_event("e2", "run", "TYPE_B", {"n": 2})
    store.save_event("e3", "run", "TYPE_A", {"n": 3})

    only_a = store.get_events("run", event_type="TYPE_A")
    assert len(only_a) == 2
    assert all(e["event_type"] == "TYPE_A" for e in only_a)
    # pagination
    assert len(store.get_events("run", limit=1, offset=1)) == 1


def test_save_event_rollback_on_error(failing_session_store) -> None:
    store, fake = failing_session_store
    with pytest.raises(RuntimeError):
        store.save_event("e-err", "run", "T", {"a": 1})
    fake.rollback.assert_called_once()
    fake.close.assert_called_once()


# ---------------------------------------------------------------------------
# Delete (cascade) + error branches
# ---------------------------------------------------------------------------


def test_delete_thread_cascades(store: AGUIPersistence) -> None:
    store.save_thread("th", user_id="u")
    store.save_run_start("run", "th")
    store.save_message("m", "th", role="user", content="x", run_id="run")
    store.save_event("e", "run", "T", {"a": 1})

    store.delete_thread("th")

    assert store.get_thread("th") is None
    # ORM cascade removed dependent rows
    assert store.get_runs("th") == []
    assert store.get_messages("th") == []
    assert store.get_events("run") == []


def test_delete_thread_not_found_logs_warning(store: AGUIPersistence) -> None:
    # No exception; hits "thread not found for deletion" branch
    store.delete_thread("ghost")
    assert store.get_thread("ghost") is None


def test_delete_thread_rollback_on_error(store: AGUIPersistence) -> None:
    store.save_thread("th", user_id="u")
    fake = MagicMock()
    # first() returns truthy mock -> enters found branch; delete raises
    fake.delete.side_effect = RuntimeError("boom")
    store._get_session = lambda: fake  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        store.delete_thread("th")
    fake.rollback.assert_called_once()
    fake.close.assert_called_once()


# ---------------------------------------------------------------------------
# _run_to_dict serialization directly (both populated + null-date branches)
# ---------------------------------------------------------------------------


def test_run_to_dict_with_null_dates(store: AGUIPersistence) -> None:
    run = Run(
        id="rx",
        thread_id="tx",
        status="running",
        error_message=None,
        metadata_json=None,
    )
    # created_at/finished_at are None (not flushed) -> exercises the else branches
    d = store._run_to_dict(run)
    assert d["id"] == "rx"
    assert d["created_at"] is None
    assert d["finished_at"] is None
    assert d["metadata"] == {}


def test_run_to_dict_with_metadata(store: AGUIPersistence) -> None:
    run = Run(
        id="ry",
        thread_id="ty",
        status="completed",
        error_message="e",
        metadata_json=json.dumps({"a": 1}),
    )
    d = store._run_to_dict(run)
    assert d["metadata"] == {"a": 1}
    assert d["error_message"] == "e"


# ---------------------------------------------------------------------------
# ORM model smoke (import + tablenames) to ensure model definitions execute
# ---------------------------------------------------------------------------


def test_model_tablenames() -> None:
    assert Thread.__tablename__ == "threads"
    assert Run.__tablename__ == "runs"
    assert Message.__tablename__ == "messages"
    assert Event.__tablename__ == "events"
