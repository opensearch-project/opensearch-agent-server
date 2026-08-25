# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for utils.obo_context.OboAuth.

Exercises token set/get, per-instance isolation, and both the sync and async
httpx auth flows (token present vs absent). Fully hermetic: builds real
httpx.Request objects locally; no network is performed.
"""

from __future__ import annotations

import httpx
import pytest

from utils.obo_context import OboAuth

pytestmark = pytest.mark.unit


def _req() -> httpx.Request:
    return httpx.Request("GET", "https://example.invalid/tool")


class TestTokenSetGet:
    def test_default_token_is_none(self):
        assert OboAuth().get_token() is None

    def test_set_then_get(self):
        auth = OboAuth()
        auth.set_token("eyJhbG.xyz")
        assert auth.get_token() == "eyJhbG.xyz"

    def test_set_none_clears(self):
        auth = OboAuth()
        auth.set_token("tok")
        auth.set_token(None)
        assert auth.get_token() is None

    def test_instances_are_isolated(self):
        a, b = OboAuth(), OboAuth()
        a.set_token("token-a")
        b.set_token("token-b")
        assert a.get_token() == "token-a"
        assert b.get_token() == "token-b"


class TestSyncAuthFlow:
    def test_injects_bearer_when_token_present(self):
        auth = OboAuth()
        auth.set_token("abc123")
        req = _req()
        gen = auth.sync_auth_flow(req)
        out = next(gen)
        assert out.headers["Authorization"] == "Bearer abc123"
        with pytest.raises(StopIteration):
            next(gen)

    def test_no_header_when_token_absent(self):
        auth = OboAuth()
        req = _req()
        out = next(auth.sync_auth_flow(req))
        assert "Authorization" not in out.headers


class TestAsyncAuthFlow:
    async def test_injects_bearer_when_token_present(self):
        auth = OboAuth()
        auth.set_token("async-tok")
        req = _req()
        gen = auth.async_auth_flow(req)
        out = await gen.__anext__()
        assert out.headers["Authorization"] == "Bearer async-tok"
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    async def test_no_header_when_token_absent(self):
        auth = OboAuth()
        req = _req()
        gen = auth.async_auth_flow(req)
        out = await gen.__anext__()
        assert "Authorization" not in out.headers
