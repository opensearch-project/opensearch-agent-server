# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.authorization.

Exercises the shared auth/request/persistence pre-check, thread and run
ownership decision branches (allow / deny / not-found / no-thread-id), the
config resolution helper, and the require_ownership decorator (thread, run,
positional + keyword extraction, resource caching, auth-disabled, and
unrecognized resource types).

Hermetic: fake Request objects (SimpleNamespace) and a fake persistence
double are used; no network, no FastAPI app, no real config singleton.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from server.authorization import (
    _get_config_from_request,
    _require_auth_request_and_persistence,
    require_ownership,
    require_run_ownership,
    require_thread_ownership,
)
from server.exceptions import (
    ForbiddenError,
    NotFoundError,
    PersistenceNotEnabledError,
    UnauthorizedError,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_request(
    *,
    auth_enabled: bool = True,
    authenticated: bool = True,
    user_id: str = "user-1",
    config_on_app: bool = True,
):
    """Build a fake FastAPI Request with app.state.config and request.state."""
    config = SimpleNamespace(auth_enabled=auth_enabled)
    app_state = SimpleNamespace(config=config if config_on_app else None)
    app = SimpleNamespace(state=app_state)
    state = SimpleNamespace(authenticated=authenticated, user_id=user_id)
    return SimpleNamespace(
        app=app,
        state=state,
        headers={},
        client=SimpleNamespace(host="1.2.3.4"),
        url=SimpleNamespace(path="/threads"),
    )


class FakePersistence:
    def __init__(
        self,
        *,
        thread=None,
        run_owned=None,
        existing_run=None,
        run_thread=None,
    ):
        self._thread = thread
        self._run_owned = run_owned
        self._existing_run = existing_run
        self._run_thread = run_thread

    def get_thread(self, thread_id):
        return self._thread if self._run_thread is None else self._run_thread

    def get_run_with_ownership_check(self, run_id, user_id):
        return self._run_owned

    def get_run(self, run_id):
        return self._existing_run


# ---------------------------------------------------------------------------
# _get_config_from_request
# ---------------------------------------------------------------------------


class TestGetConfigFromRequest:
    def test_returns_app_state_config(self):
        req = _make_request(auth_enabled=False)
        cfg = _get_config_from_request(req)
        assert cfg.auth_enabled is False

    def test_request_none_falls_back_to_get_config(self):
        sentinel = SimpleNamespace(auth_enabled=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("server.config.get_config", lambda: sentinel, raising=True)
            assert _get_config_from_request(None) is sentinel

    def test_app_state_config_none_falls_back(self):
        req = _make_request(config_on_app=False)
        sentinel = SimpleNamespace(auth_enabled=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("server.config.get_config", lambda: sentinel, raising=True)
            assert _get_config_from_request(req) is sentinel


# ---------------------------------------------------------------------------
# _require_auth_request_and_persistence
# ---------------------------------------------------------------------------


class TestRequireAuthRequestAndPersistence:
    def test_auth_disabled_returns_false(self):
        req = _make_request(auth_enabled=False)
        assert (
            _require_auth_request_and_persistence(
                req, FakePersistence(), None, "op", "threadId", "t1"
            )
            is False
        )

    def test_no_request_raises_forbidden(self):
        # request None -> config resolves via get_config(); patch it auth-enabled
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "server.config.get_config",
                lambda: SimpleNamespace(auth_enabled=True),
                raising=True,
            )
            with pytest.raises(ForbiddenError):
                _require_auth_request_and_persistence(
                    None, FakePersistence(), None, "op", "threadId", "t1"
                )

    def test_not_authenticated_raises_unauthorized(self):
        req = _make_request(authenticated=False)
        with pytest.raises(UnauthorizedError):
            _require_auth_request_and_persistence(
                req, FakePersistence(), None, "op", "threadId", "t1"
            )

    def test_no_persistence_raises_persistence_not_enabled(self):
        req = _make_request()
        with pytest.raises(PersistenceNotEnabledError):
            _require_auth_request_and_persistence(
                req, None, {"k": "v"}, "op", "threadId", "t1"
            )

    def test_all_ok_returns_true(self):
        req = _make_request()
        assert (
            _require_auth_request_and_persistence(
                req, FakePersistence(), None, "op", "threadId", "t1"
            )
            is True
        )


# ---------------------------------------------------------------------------
# require_thread_ownership
# ---------------------------------------------------------------------------


class TestRequireThreadOwnership:
    def test_auth_disabled_returns_none(self):
        req = _make_request(auth_enabled=False)
        assert require_thread_ownership(FakePersistence(), "t1", req) is None

    def test_owner_match_returns_thread(self):
        req = _make_request(user_id="user-1")
        p = FakePersistence(thread={"user_id": "user-1", "id": "t1"})
        result = require_thread_ownership(p, "t1", req)
        assert result == {"user_id": "user-1", "id": "t1"}

    def test_owner_mismatch_raises_forbidden(self):
        req = _make_request(user_id="user-1")
        p = FakePersistence(thread={"user_id": "someone-else"})
        with pytest.raises(ForbiddenError):
            require_thread_ownership(p, "t1", req, context={"x": 1})

    def test_missing_thread_returns_none(self):
        req = _make_request(user_id="user-1")
        p = FakePersistence(thread=None)
        assert require_thread_ownership(p, "t1", req) is None


# ---------------------------------------------------------------------------
# require_run_ownership
# ---------------------------------------------------------------------------


class TestRequireRunOwnership:
    def test_auth_disabled_returns_none(self):
        req = _make_request(auth_enabled=False)
        assert require_run_ownership(FakePersistence(), "r1", req) is None

    def test_happy_path_single_query(self):
        req = _make_request()
        run = {"id": "r1", "thread_id": "t1"}
        p = FakePersistence(run_owned=run)
        assert require_run_ownership(p, "r1", req) == run

    def test_run_not_found_raises_notfound(self):
        req = _make_request()
        p = FakePersistence(run_owned=None, existing_run=None)
        with pytest.raises(NotFoundError):
            require_run_ownership(p, "r1", req)

    def test_run_without_thread_id_returns_run(self):
        req = _make_request()
        run = {"id": "r1"}  # no thread_id / threadId
        p = FakePersistence(run_owned=None, existing_run=run)
        assert require_run_ownership(p, "r1", req) == run

    def test_thread_owned_returns_run(self):
        req = _make_request(user_id="user-1")
        existing = {"id": "r1", "thread_id": "t1"}
        p = FakePersistence(
            run_owned=None,
            existing_run=existing,
            run_thread={"user_id": "user-1"},
        )
        assert require_run_ownership(p, "r1", req) == existing

    def test_thread_not_owned_raises_forbidden(self):
        req = _make_request(user_id="user-1")
        existing = {"id": "r1", "threadId": "t1"}
        p = FakePersistence(
            run_owned=None,
            existing_run=existing,
            run_thread={"user_id": "other"},
        )
        with pytest.raises(ForbiddenError):
            require_run_ownership(p, "r1", req)

    def test_thread_missing_raises_forbidden(self):
        # existing run has thread_id but get_thread returns None -> forbidden path
        req = _make_request(user_id="user-1")
        existing = {"id": "r1", "thread_id": "t1"}
        p = FakePersistence(
            run_owned=None,
            existing_run=existing,
            run_thread=False,  # falsy but not None -> get_thread returns False
        )
        with pytest.raises(ForbiddenError):
            require_run_ownership(p, "r1", req)


# ---------------------------------------------------------------------------
# require_ownership decorator
# ---------------------------------------------------------------------------


class TestRequireOwnershipDecorator:
    def test_thread_kwargs_caches_resource(self):
        req = _make_request(user_id="user-1")
        thread = {"user_id": "user-1", "id": "t1"}
        p = FakePersistence(thread=thread)

        @require_ownership("thread", "thread_id")
        def route(persistence, thread_id, request, _cached_thread=None):
            return _cached_thread

        result = route(persistence=p, thread_id="t1", request=req)
        assert result == thread

    def test_run_positional_args(self):
        req = _make_request()
        run = {"id": "r1", "thread_id": "t1"}
        p = FakePersistence(run_owned=run)

        @require_ownership("run")  # default id_param -> run_id
        def route(persistence, run_id, request, _cached_run=None):
            return _cached_run

        # positional: persistence, run_id, request
        assert route(p, "r1", req) == run

    def test_auth_disabled_no_cache(self):
        req = _make_request(auth_enabled=False)
        p = FakePersistence(thread={"user_id": "x"})

        @require_ownership("thread", "thread_id")
        def route(persistence, thread_id, request, _cached_thread="default"):
            return _cached_thread

        # ownership returns None (auth disabled) -> cache not injected -> default kept
        assert route(persistence=p, thread_id="t1", request=req) == "default"

    def test_unrecognized_resource_type_skips_check(self):
        req = _make_request()
        p = FakePersistence()

        @require_ownership("widget", "widget_id")
        def route(persistence, widget_id, request):
            return "ran"

        assert route(persistence=p, widget_id="w1", request=req) == "ran"

    def test_missing_params_skips_check(self):
        # resource_id absent -> ownership check block skipped, fn still runs
        @require_ownership("thread", "thread_id")
        def route(persistence=None, thread_id=None, request=None):
            return "no-check"

        assert route() == "no-check"
