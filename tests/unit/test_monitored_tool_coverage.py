# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for utils.monitored_tool.

Covers the sync/async wrappers, graceful degradation, the AG-UI emitter path,
the ``inputSchema`` branch, and the ``_result_preview`` helper.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.monitored_tool import _result_preview, monitored_tool

pytestmark = pytest.mark.unit

_AG_UI_PATCH = "utils.monitored_tool.get_ag_ui_emitter"
_AG_UI_AVAILABLE_PATCH = "utils.monitored_tool.GET_AG_UI_EMITTER_AVAILABLE"


class _AsyncCM:
    def __init__(self):
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return "tool-call-id"

    async def __aexit__(self, *args):
        self.exited = True
        return None


class TestResultPreview:
    def test_short_result(self):
        assert _result_preview("hi") == "hi"

    def test_long_result_truncated(self):
        out = _result_preview("x" * 500, max_len=200)
        assert out.endswith("...")
        assert len(out) == 203

    def test_unstringifiable_result(self):
        class Bad:
            def __str__(self):
                raise RuntimeError("no str")

        assert _result_preview(Bad()) == "[output omitted]"


class TestGracefulDegradation:
    async def test_async_no_emitter(self):
        with patch(_AG_UI_PATCH, return_value=None):

            @monitored_tool(name="T", description="d")
            async def _tool(x: int) -> int:
                return x + 1

            assert await _tool(1) == 2

    async def test_sync_no_emitter(self):
        with patch(_AG_UI_PATCH, return_value=None):

            @monitored_tool(name="T", description="d")
            def _tool(x: int) -> int:
                return x * 3

            assert await _tool(2) == 6

    async def test_emitter_getter_raises(self):
        with (
            patch(_AG_UI_PATCH, side_effect=RuntimeError("no ctx")),
            patch(_AG_UI_AVAILABLE_PATCH, True),
        ):

            @monitored_tool(name="T", description="d")
            async def _tool() -> str:
                return "ok"

            assert await _tool() == "ok"

    async def test_sync_emitter_getter_raises(self):
        with (
            patch(_AG_UI_PATCH, side_effect=AttributeError("no ctx")),
            patch(_AG_UI_AVAILABLE_PATCH, True),
        ):

            @monitored_tool(name="T", description="d")
            def _tool() -> str:
                return "sync-ok"

            assert await _tool() == "sync-ok"


class TestAgUiEmitterPath:
    async def test_async_uses_emitter(self):
        acm = _AsyncCM()
        emitter = MagicMock()
        emitter.tool_call.return_value = acm
        emitter.set_tool_call_result = AsyncMock(return_value=None)
        with (
            patch(_AG_UI_PATCH, return_value=emitter),
            patch(_AG_UI_AVAILABLE_PATCH, True),
        ):

            @monitored_tool(name="AgUi", description="d")
            async def _tool() -> str:
                return "done"

            assert await _tool() == "done"
        assert acm.entered and acm.exited
        emitter.tool_call.assert_called_once_with("AgUi")
        emitter.set_tool_call_result.assert_awaited_once()

    async def test_sync_uses_emitter(self):
        acm = _AsyncCM()
        emitter = MagicMock()
        emitter.tool_call.return_value = acm
        emitter.set_tool_call_result = AsyncMock(return_value=None)
        with (
            patch(_AG_UI_PATCH, return_value=emitter),
            patch(_AG_UI_AVAILABLE_PATCH, True),
        ):

            @monitored_tool(name="SyncAgUi", description="d")
            def _tool() -> str:
                return "sync"

            assert await _tool() == "sync"
        emitter.tool_call.assert_called_once_with("SyncAgUi")

    async def test_name_falls_back_to_func_name(self):
        acm = _AsyncCM()
        emitter = MagicMock()
        emitter.tool_call.return_value = acm
        emitter.set_tool_call_result = AsyncMock(return_value=None)
        with (
            patch(_AG_UI_PATCH, return_value=emitter),
            patch(_AG_UI_AVAILABLE_PATCH, True),
        ):

            @monitored_tool(description="d")
            async def my_named_tool() -> str:
                return "ok"

            await my_named_tool()
        emitter.tool_call.assert_called_once_with("my_named_tool")


class TestInputSchemaBranch:
    async def test_async_with_input_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        with patch(_AG_UI_PATCH, return_value=None):

            @monitored_tool(name="Schemad", description="d", inputSchema=schema)
            async def _tool(x: int) -> int:
                return x

            assert await _tool(5) == 5

    async def test_sync_with_input_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        with patch(_AG_UI_PATCH, return_value=None):

            @monitored_tool(name="SchemadSync", description="d", inputSchema=schema)
            def _tool(x: int) -> int:
                return x

            assert await _tool(7) == 7


class TestErrorPropagation:
    async def test_async_error_propagates(self):
        with patch(_AG_UI_PATCH, return_value=None):

            @monitored_tool(name="Raises", description="d")
            async def _tool() -> str:
                raise ValueError("fail")

            with pytest.raises(ValueError, match="fail"):
                await _tool()

    async def test_sync_error_propagates(self):
        with patch(_AG_UI_PATCH, return_value=None):

            @monitored_tool(name="SyncRaises", description="d")
            def _tool() -> str:
                raise RuntimeError("boom")

            with pytest.raises(RuntimeError, match="boom"):
                await _tool()
