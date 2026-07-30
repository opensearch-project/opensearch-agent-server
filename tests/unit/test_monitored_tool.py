"""
Unit tests for monitored_tool decorator.

Tests the monitoring wrapper, graceful degradation when the AG-UI
context is unavailable, and AG-UI event emission when the emitter is available.
Covers both async and sync tool functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.monitored_tool import monitored_tool

pytestmark = pytest.mark.unit


class _AsyncContextManagerMock:
    """Async context manager for mocking ag_ui_emitter.tool_call."""

    def __init__(self, step=None):
        self.step = MagicMock() if step is None else step
        if not hasattr(self.step, "output"):
            self.step.output = ""
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self.step

    async def __aexit__(self, *args):
        self.exited = True
        return None


# Patches used by all tests: patch get_ag_ui_emitter at point of use.
# GET_AG_UI_EMITTER_AVAILABLE is set at import; if ag_ui_tool_event_emitter fails to
# import it is False. Tests that need the AG-UI path must also patch it True.
_AG_UI_PATCH = "utils.monitored_tool.get_ag_ui_emitter"
_AG_UI_AVAILABLE_PATCH = "utils.monitored_tool.GET_AG_UI_EMITTER_AVAILABLE"


class TestMonitoredToolGracefulDegradation:
    """Tests for graceful degradation when the AG-UI emitter is unavailable."""

    @pytest.mark.asyncio
    async def test_async_tool_executes_without_monitoring_when_emitter_returns_none(
        self,
    ):
        """Tool runs and returns result when get_ag_ui_emitter returns None."""
        with patch(_AG_UI_PATCH, return_value=None):

            @monitored_tool(name="NoMonitorTool", description="Test")
            async def _tool(x: int) -> int:
                return x + 1

            result = await _tool(41)
            assert result == 42

    @pytest.mark.asyncio
    async def test_async_tool_executes_when_get_ag_ui_emitter_raises(self):
        """Tool runs when get_ag_ui_emitter raises; degradation to no monitoring."""
        with patch(_AG_UI_PATCH, side_effect=ImportError("No AG-UI")):

            @monitored_tool(name="AgUiRaises", description="Test")
            async def _tool() -> str:
                return "ok"

            result = await _tool()
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_sync_tool_executes_without_monitoring(self):
        """Sync tool runs and returns result when AG-UI is not available."""
        with patch(_AG_UI_PATCH, return_value=None):

            @monitored_tool(name="SyncNoMonitor", description="Test")
            def _tool(x: int) -> int:
                return x * 2

            result = await _tool(21)
            assert result == 42


class TestMonitoredToolAgUiEmission:
    """Tests for AG-UI tool event emission."""

    @pytest.mark.asyncio
    async def test_async_tool_uses_ag_ui_tool_call_when_available(self):
        """When AG-UI emitter is available, ag_ui_emitter.tool_call is used."""
        acm = _AsyncContextManagerMock()
        mock_emitter = MagicMock()
        mock_emitter.tool_call.return_value = acm
        mock_emitter.set_tool_call_result = AsyncMock(return_value=None)

        with (
            patch(_AG_UI_PATCH, return_value=mock_emitter),
            patch(_AG_UI_AVAILABLE_PATCH, True),
        ):

            @monitored_tool(name="AgUiOnlyTool", description="Test")
            async def _tool() -> str:
                return "done"

            result = await _tool()

        assert result == "done"
        assert acm.entered and acm.exited
        mock_emitter.tool_call.assert_called_once_with("AgUiOnlyTool")

    @pytest.mark.asyncio
    async def test_sync_tool_uses_ag_ui_tool_call_when_available(self):
        """Sync tool uses ag_ui_emitter.tool_call when available."""
        acm = _AsyncContextManagerMock()
        mock_emitter = MagicMock()
        mock_emitter.tool_call.return_value = acm
        mock_emitter.set_tool_call_result = AsyncMock(return_value=None)

        with (
            patch(_AG_UI_PATCH, return_value=mock_emitter),
            patch(_AG_UI_AVAILABLE_PATCH, True),
        ):

            @monitored_tool(name="SyncAgUi", description="Test")
            def _tool() -> str:
                return "sync-agui"

            result = await _tool()

        assert result == "sync-agui"
        mock_emitter.tool_call.assert_called_once_with("SyncAgUi")


class TestMonitoredToolWrapperBehavior:
    """Tests for decorator and wrapper behavior (name, args, errors)."""

    @pytest.mark.asyncio
    async def test_tool_name_from_decorator_when_provided(self):
        """Explicit name= is passed to emitter tool_call."""
        acm = _AsyncContextManagerMock()
        mock_emitter = MagicMock()
        mock_emitter.tool_call.return_value = acm
        mock_emitter.set_tool_call_result = AsyncMock(return_value=None)

        with (
            patch(_AG_UI_PATCH, return_value=mock_emitter),
            patch(_AG_UI_AVAILABLE_PATCH, True),
        ):

            @monitored_tool(name="ExplicitName", description="Test")
            async def _impl() -> str:
                return "ok"

            await _impl()

        mock_emitter.tool_call.assert_called_once_with("ExplicitName")

    @pytest.mark.asyncio
    async def test_tool_name_falls_back_to_func_name(self):
        """When name= is not provided, func.__name__ is used."""
        acm = _AsyncContextManagerMock()
        mock_emitter = MagicMock()
        mock_emitter.tool_call.return_value = acm
        mock_emitter.set_tool_call_result = AsyncMock(return_value=None)

        with (
            patch(_AG_UI_PATCH, return_value=mock_emitter),
            patch(_AG_UI_AVAILABLE_PATCH, True),
        ):

            @monitored_tool(description="Test")
            async def my_custom_tool() -> str:
                return "ok"

            await my_custom_tool()

        mock_emitter.tool_call.assert_called_once_with("my_custom_tool")

    @pytest.mark.asyncio
    async def test_async_tool_error_propagates(self):
        """When the underlying async tool raises, the exception propagates to the caller."""
        with patch(_AG_UI_PATCH, return_value=None):

            @monitored_tool(name="RaisesTool", description="Test")
            async def _tool() -> str:
                raise ValueError("tool failed")

            with pytest.raises(ValueError, match="tool failed"):
                await _tool()

    @pytest.mark.asyncio
    async def test_sync_tool_error_propagates(self):
        """When the underlying sync tool raises, the exception propagates to the caller."""
        with patch(_AG_UI_PATCH, return_value=None):

            @monitored_tool(name="SyncRaises", description="Test")
            def _tool() -> str:
                raise RuntimeError("sync failed")

            with pytest.raises(RuntimeError, match="sync failed"):
                await _tool()

    @pytest.mark.asyncio
    async def test_async_tool_error_propagates_when_ag_ui_used(self):
        """When AG-UI emitter is used and the tool raises, the exception propagates."""
        acm = _AsyncContextManagerMock()
        mock_emitter = MagicMock()
        mock_emitter.tool_call.return_value = acm

        with (
            patch(_AG_UI_PATCH, return_value=mock_emitter),
            patch(_AG_UI_AVAILABLE_PATCH, True),
        ):

            @monitored_tool(name="AgUiRaises", description="Test")
            async def _tool() -> str:
                raise TypeError("type")

            with pytest.raises(TypeError, match="type"):
                await _tool()
