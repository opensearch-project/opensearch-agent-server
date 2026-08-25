# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.rate_limiting."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI

from server.rate_limiting import (
    create_rate_limiter,
    get_rate_limit_decorator,
    get_rate_limit_key,
    setup_rate_limiting,
)

pytestmark = pytest.mark.unit


class _Headers:
    """Case-insensitive header shim matching starlette Headers.get semantics."""

    def __init__(self, data: dict[str, str]):
        self._data = {k.lower(): v for k, v in data.items()}

    def get(self, key: str, default=None):
        return self._data.get(key.lower(), default)


def _make_request(headers=None, user_id=None, client_host="10.0.0.1"):
    state = SimpleNamespace()
    if user_id is not None:
        state.user_id = user_id
    client = SimpleNamespace(host=client_host) if client_host else None
    return SimpleNamespace(
        headers=_Headers(headers or {}),
        state=state,
        client=client,
    )


def _cfg(enabled=True, per_minute=60, per_hour=1000):
    return SimpleNamespace(
        rate_limit_enabled=enabled,
        rate_limit_per_minute=per_minute,
        rate_limit_per_hour=per_hour,
    )


class TestGetRateLimitKey:
    def test_uses_user_id_when_present(self):
        """A user id distinct from the client IP produces a user: key."""
        req = _make_request(headers={"X-User-Id": "alice"}, client_host="10.0.0.1")
        assert get_rate_limit_key(req) == "user:alice"

    def test_falls_back_to_ip_when_user_id_equals_remote(self):
        """When the resolved user id equals the remote address, the IP is used."""
        # request.state.user_id short-circuits get_user_id_from_request, so we can
        # force it to equal the client host and hit the IP fallback branch.
        req = _make_request(user_id="10.0.0.1", client_host="10.0.0.1")
        assert get_rate_limit_key(req) == "10.0.0.1"


class TestCreateRateLimiter:
    def test_disabled_returns_none(self):
        assert create_rate_limiter(_cfg(enabled=False)) is None

    def test_enabled_returns_limiter(self):
        limiter = create_rate_limiter(_cfg(enabled=True, per_minute=30, per_hour=500))
        assert limiter is not None
        assert hasattr(limiter, "limit")


class TestSetupRateLimiting:
    def test_none_limiter_is_noop(self):
        app = FastAPI()
        setup_rate_limiting(app, None)
        assert getattr(app.state, "limiter", None) is None

    def test_attaches_limiter_and_handler(self):
        app = FastAPI()
        limiter = MagicMock()
        setup_rate_limiting(app, limiter)
        assert app.state.limiter is limiter


class TestGetRateLimitDecorator:
    def test_none_limiter_returns_noop_decorator(self):
        decorator = get_rate_limit_decorator(None)

        def handler():
            return "ok"

        assert decorator(handler) is handler

    def test_per_minute_only(self):
        limiter = MagicMock()
        get_rate_limit_decorator(limiter, per_minute=5)
        limiter.limit.assert_called_once_with("5/minute")

    def test_per_hour_only(self):
        limiter = MagicMock()
        get_rate_limit_decorator(limiter, per_hour=100)
        limiter.limit.assert_called_once_with("100/hour")

    def test_per_minute_and_per_hour(self):
        limiter = MagicMock()
        get_rate_limit_decorator(limiter, per_minute=5, per_hour=100)
        limiter.limit.assert_called_once_with("5/minute, 100/hour")

    def test_defaults_from_config_when_no_overrides(self):
        limiter = MagicMock()
        get_rate_limit_decorator(limiter, config=_cfg(per_minute=60, per_hour=1000))
        limiter.limit.assert_called_once_with("60/minute, 1000/hour")
