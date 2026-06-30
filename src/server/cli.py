"""CLI entry point for opensearch-agent-server.

Provides the ``opensearch-agent-server`` console command installed by pip.

This module mirrors the bootstrapping logic in ``run_server.py`` (dotenv loading,
AWS credential bridging, logging configuration) so the server behaves identically
whether started via ``python run_server.py`` or the installed CLI command.

Usage::

    # Start agent server only
    opensearch-agent-server

    # Start agent server + MCP server together
    opensearch-agent-server --with-mcp

    # Custom MCP port and config
    opensearch-agent-server --with-mcp --mcp-port 3002 --mcp-config ./my_config.yml
"""

from __future__ import annotations

import argparse
import atexit
import configparser
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Default MCP config bundled with the package.
_DEFAULT_MCP_CONFIG = Path(__file__).parent / "default_mcp_config.yml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="opensearch-agent-server",
        description="OpenSearch Agent Server — multi-agent orchestrator with AG-UI protocol.",
    )
    parser.add_argument(
        "--with-mcp",
        action="store_true",
        default=False,
        help="Also start the OpenSearch MCP Server as a background process.",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=int(os.getenv("MCP_SERVER_PORT", "3001")),
        help="Port for the MCP server (default: 3001). Only used with --with-mcp.",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP server config YAML. Defaults to bundled config.",
    )
    return parser.parse_args(argv)


def _bridge_aws_credentials() -> None:
    """Bridge AWS credentials from ~/.aws/credentials to environment variables.

    ART's agent_config.py reads AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from
    ``os.getenv()`` rather than using the default boto3 credential chain.
    When the env vars are not set, this function reads them from the AWS
    credentials file so that Bedrock-backed agents work out of the box.
    """
    if os.getenv("AWS_ACCESS_KEY_ID"):
        return

    cred_file = os.path.expanduser("~/.aws/credentials")
    if not os.path.isfile(cred_file):
        return

    cp = configparser.ConfigParser(strict=False)
    cp.read(cred_file)
    profile = os.getenv("AWS_PROFILE", "default")

    if not cp.has_section(profile):
        return

    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        val = cp.get(profile, key, fallback=None) or cp.get(
            profile, key.lower(), fallback=None
        )
        if val and not os.getenv(key):
            os.environ[key] = val.strip()


def _wait_for_port(port: int, timeout: int = 30) -> bool:
    """Wait for a TCP port to become available.

    Returns True if the port is ready, False if timeout reached.
    """
    import socket

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _start_mcp_server(port: int, config_path: str | None) -> subprocess.Popen:
    """Start the OpenSearch MCP Server as a subprocess.

    Args:
        port: Port for the MCP server to listen on.
        config_path: Path to MCP config YAML, or None for bundled default.

    Returns:
        The subprocess.Popen object for the MCP server.

    Raises:
        SystemExit: If the MCP server command is not found or fails to start.
    """
    mcp_cmd = shutil.which("opensearch-mcp-server-py")
    if mcp_cmd is None:
        print(
            "Error: opensearch-mcp-server-py command not found.\n"
            "Install it with: pip install opensearch-mcp-server-py",
            file=sys.stderr,
        )
        sys.exit(1)

    config = config_path or str(_DEFAULT_MCP_CONFIG)
    if not Path(config).is_file():
        print(f"Error: MCP config file not found: {config}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        mcp_cmd,
        "--transport",
        "stream",
        "--port",
        str(port),
        "--config",
        config,
    ]

    env = {
        **os.environ,
        "OPENSEARCH_HEADER_AUTH": "true",
    }

    print(f"Starting MCP Server on localhost:{port}...")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Register cleanup so MCP server stops when agent server stops.
    def _cleanup_mcp() -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    atexit.register(_cleanup_mcp)
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))

    # Wait for MCP server to be ready.
    if not _wait_for_port(port, timeout=30):
        print(
            f"Error: MCP Server failed to start on port {port} within 30s.",
            file=sys.stderr,
        )
        _cleanup_mcp()
        sys.exit(1)

    print(f"MCP Server ready on localhost:{port}")
    return process


def main(argv: list[str] | None = None) -> None:
    """Start the OpenSearch Agent Server.

    This is the entry point registered as the ``opensearch-agent-server``
    console script.  It performs the same bootstrap sequence as
    ``run_server.py``:

    1. Load ``.env`` file (must happen before any server imports that read config).
    2. Bridge AWS credentials from ``~/.aws/credentials``.
    3. Optionally start the MCP server (``--with-mcp``).
    4. Configure structured logging (JSON or human-readable).
    5. Import the FastAPI app and start Uvicorn.
    """
    args = _parse_args(argv)

    # 1. Load .env BEFORE any server imports that instantiate config at import time.
    load_dotenv()

    # 2. Bridge AWS credentials for Bedrock-backed agents.
    _bridge_aws_credentials()

    # 3. Optionally start MCP server.
    if args.with_mcp:
        _start_mcp_server(args.mcp_port, args.mcp_config)
        # Tell the agent server where to find MCP.
        os.environ.setdefault("MCP_SERVER_URL", f"http://localhost:{args.mcp_port}/mcp")

    # 4. Configure logging (imports must come after load_dotenv).
    from server.logging_config import configure_logging, get_logging_config

    use_json, log_level = get_logging_config()
    configure_logging(use_json=use_json, log_level=log_level, force=True)

    # 5. Import app and config after environment is fully set up.
    from server.ag_ui_app import app
    from server.config import get_config
    from utils.logging_helpers import get_logger, log_info_event

    logger = get_logger(__name__)
    config = get_config()
    host = config.server_host
    port = config.server_port

    log_info_event(
        logger,
        f"Starting OpenSearch Agent Server on {host}:{port}",
        "server.starting",
        host=host,
        port=port,
    )
    log_info_event(
        logger,
        f"  POST /runs    -> http://{host}:{port}/runs",
        "server.endpoint",
    )
    log_info_event(
        logger,
        f"  GET  /agents  -> http://{host}:{port}/agents",
        "server.agents_endpoint",
    )
    log_info_event(
        logger,
        f"  GET  /health  -> http://{host}:{port}/health",
        "server.health_endpoint",
    )

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
