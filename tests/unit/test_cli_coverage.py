# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for ``server.cli``.

Everything with a side effect is mocked: ``uvicorn.run`` never starts a server,
``subprocess.Popen`` never spawns a process, ``socket.create_connection`` never
binds a port, and the deferred server imports inside ``main()`` are replaced
with fake modules.  All tests are hermetic and deterministic.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from server import cli

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------
class TestParseArgs:
    def test_defaults(self) -> None:
        ns = cli._parse_args([])
        assert ns.with_mcp is False
        assert ns.mcp_port == 3001
        assert ns.mcp_config is None

    def test_all_flags(self) -> None:
        ns = cli._parse_args(
            ["--with-mcp", "--mcp-port", "3002", "--mcp-config", "/tmp/x.yml"]
        )
        assert ns.with_mcp is True
        assert ns.mcp_port == 3002
        assert ns.mcp_config == "/tmp/x.yml"


# ---------------------------------------------------------------------------
# _bridge_aws_credentials
# ---------------------------------------------------------------------------
class TestBridgeAwsCredentials:
    def test_early_return_when_key_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "already-set")
        # Should short-circuit before touching the filesystem.
        with patch.object(cli.os.path, "isfile") as isfile:
            cli._bridge_aws_credentials()
        isfile.assert_not_called()

    def test_no_credentials_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.setattr(cli.os.path, "expanduser", lambda p: "/no/such/file")
        # Simply returns without raising.
        cli._bridge_aws_credentials()

    def test_profile_section_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        cred = tmp_path / "credentials"
        cred.write_text("[other]\naws_access_key_id = abc\n")
        monkeypatch.setattr(cli.os.path, "expanduser", lambda p: str(cred))
        monkeypatch.setenv("AWS_PROFILE", "default")
        cli._bridge_aws_credentials()
        assert "AWS_ACCESS_KEY_ID" not in cli.os.environ

    def test_bridges_keys_from_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        cred = tmp_path / "credentials"
        cred.write_text(
            "[default]\n"
            "aws_access_key_id = AKIA_TEST\n"
            "aws_secret_access_key = SECRET_TEST\n"
            "aws_session_token = TOKEN_TEST\n"
        )
        monkeypatch.setattr(cli.os.path, "expanduser", lambda p: str(cred))
        monkeypatch.setenv("AWS_PROFILE", "default")
        try:
            cli._bridge_aws_credentials()
            assert cli.os.environ["AWS_ACCESS_KEY_ID"] == "AKIA_TEST"
            assert cli.os.environ["AWS_SECRET_ACCESS_KEY"] == "SECRET_TEST"
            assert cli.os.environ["AWS_SESSION_TOKEN"] == "TOKEN_TEST"
        finally:
            for key in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
            ):
                cli.os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# _wait_for_port
# ---------------------------------------------------------------------------
class TestWaitForPort:
    def test_port_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr("socket.create_connection", lambda *a, **k: conn)
        assert cli._wait_for_port(1234, timeout=5) is True

    def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*a, **k):
            raise OSError("refused")

        monkeypatch.setattr("socket.create_connection", _raise)
        monkeypatch.setattr(cli.time, "sleep", lambda s: None)
        # start=0; first predicate 0<1 enters the loop (OSError+sleep), second
        # predicate 100 exits — exercising the except/sleep branch.
        clock = iter([0.0, 0.0, 100.0])
        monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
        assert cli._wait_for_port(1234, timeout=1) is False


# ---------------------------------------------------------------------------
# _start_mcp_server
# ---------------------------------------------------------------------------
class TestStartMcpServer:
    def test_command_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli.shutil, "which", lambda name: None)
        with pytest.raises(SystemExit) as exc:
            cli._start_mcp_server(3001, None)
        assert exc.value.code == 1

    def test_config_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/mcp")
        with pytest.raises(SystemExit) as exc:
            cli._start_mcp_server(3001, "/no/such/config.yml")
        assert exc.value.code == 1

    def test_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        cfg = tmp_path / "mcp.yml"
        cfg.write_text("servers: []\n")
        monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/mcp")
        proc = MagicMock()
        proc.poll.return_value = None
        popen = MagicMock(return_value=proc)
        monkeypatch.setattr(cli.subprocess, "Popen", popen)
        monkeypatch.setattr(cli, "_wait_for_port", lambda port, timeout=30: True)
        monkeypatch.setattr(cli.atexit, "register", MagicMock())
        monkeypatch.setattr(cli.signal, "signal", MagicMock())

        result = cli._start_mcp_server(3001, str(cfg))
        assert result is proc
        popen.assert_called_once()
        cli.atexit.register.assert_called_once()

    def test_port_start_timeout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cfg = tmp_path / "mcp.yml"
        cfg.write_text("servers: []\n")
        monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/mcp")
        proc = MagicMock()
        proc.poll.return_value = None
        # Force the cleanup path to hit the kill() fallback.
        proc.wait.side_effect = cli.subprocess.TimeoutExpired(cmd="mcp", timeout=5)
        monkeypatch.setattr(cli.subprocess, "Popen", MagicMock(return_value=proc))
        monkeypatch.setattr(cli, "_wait_for_port", lambda port, timeout=30: False)
        monkeypatch.setattr(cli.atexit, "register", MagicMock())
        monkeypatch.setattr(cli.signal, "signal", MagicMock())

        with pytest.raises(SystemExit) as exc:
            cli._start_mcp_server(3001, str(cfg))
        assert exc.value.code == 1
        # Cleanup terminates then kills the spawned process on failure.
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _fake_server_modules() -> dict[str, types.ModuleType]:
    """Build fake server-side modules imported lazily inside ``main()``."""
    lc = types.ModuleType("server.logging_config")
    lc.get_logging_config = lambda: (False, "INFO")
    lc.configure_logging = lambda **kw: None

    app_mod = types.ModuleType("server.ag_ui_app")
    app_mod.app = object()

    cfg = MagicMock()
    cfg.server_host = "127.0.0.1"
    cfg.server_port = 8080
    cfg_mod = types.ModuleType("server.config")
    cfg_mod.get_config = lambda: cfg

    lh = types.ModuleType("utils.logging_helpers")
    lh.get_logger = lambda name: MagicMock()
    lh.log_info_event = lambda *a, **k: None

    return {
        "server.logging_config": lc,
        "server.ag_ui_app": app_mod,
        "server.config": cfg_mod,
        "utils.logging_helpers": lh,
    }


class TestMain:
    def test_main_without_mcp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "load_dotenv", lambda: None)
        monkeypatch.setattr(cli, "_bridge_aws_credentials", lambda: None)
        run = MagicMock()
        monkeypatch.setattr(cli.uvicorn, "run", run)
        start_mcp = MagicMock()
        monkeypatch.setattr(cli, "_start_mcp_server", start_mcp)

        with patch.dict(sys.modules, _fake_server_modules()):
            cli.main([])

        start_mcp.assert_not_called()
        run.assert_called_once()
        assert run.call_args.kwargs == {"host": "127.0.0.1", "port": 8080}

    def test_main_with_mcp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "load_dotenv", lambda: None)
        monkeypatch.setattr(cli, "_bridge_aws_credentials", lambda: None)
        monkeypatch.delenv("MCP_SERVER_URL", raising=False)
        run = MagicMock()
        monkeypatch.setattr(cli.uvicorn, "run", run)
        start_mcp = MagicMock()
        monkeypatch.setattr(cli, "_start_mcp_server", start_mcp)

        try:
            with patch.dict(sys.modules, _fake_server_modules()):
                cli.main(["--with-mcp", "--mcp-port", "3005"])
            start_mcp.assert_called_once_with(3005, None)
            assert cli.os.environ["MCP_SERVER_URL"] == "http://localhost:3005/mcp"
            run.assert_called_once()
        finally:
            cli.os.environ.pop("MCP_SERVER_URL", None)
