"""OpenSearch Agent Server — Entry Point.

Run this server to expose the multi-agent orchestrator via AG-UI protocol.

Usage:
    python run_server.py
    # or
    uvicorn server.ag_ui_app:app --host 0.0.0.0 --port 8001
"""

# Ensure local src/ packages take priority over installed packages.
# Both opensearch-agent-server and os-art export a "server" package via editable
# installs (.pth files). The os-art .pth is processed first alphabetically,
# putting ART's src/ ahead of ours on sys.path. We must force our local src/
# to position 0 regardless of whether it's already in sys.path elsewhere.
import os
import sys

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if sys.path[0] != _src_dir:
    # Remove any existing entries first, then insert at position 0
    while _src_dir in sys.path:
        sys.path.remove(_src_dir)
    sys.path.insert(0, _src_dir)

# Load environment variables FIRST, before any server imports that read config.
# ag_ui_app.py calls create_app() at module level, which creates ServerConfig()
# from env vars. If load_dotenv() runs after that import, .env values are missed.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Bridge AWS credentials from ~/.aws/credentials to env vars.
# Only needed when using Bedrock (not Ollama).
from utils.model_factory import get_provider  # noqa: E402

if get_provider() == "bedrock" and not os.getenv("AWS_ACCESS_KEY_ID"):
    _cred_file = os.path.expanduser("~/.aws/credentials")
    if os.path.isfile(_cred_file):
        import configparser

        _cp = configparser.ConfigParser(strict=False)
        _cp.read(_cred_file)
        _profile = os.getenv("AWS_PROFILE", "default")
        if _cp.has_section(_profile):
            for key in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
            ):
                val = _cp.get(_profile, key, fallback=None) or _cp.get(
                    _profile, key.lower(), fallback=None
                )
                if val and not os.getenv(key):
                    os.environ[key] = val.strip()

import uvicorn  # noqa: E402

from server.logging_config import configure_logging, get_logging_config  # noqa: E402

# Configure logging after environment variables are loaded
use_json, log_level = get_logging_config()
configure_logging(use_json=use_json, log_level=log_level, force=True)

from server.ag_ui_app import app  # noqa: E402
from server.config import get_config  # noqa: E402
from utils.logging_helpers import get_logger, log_info_event  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """Run the OpenSearch Agent Server."""
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
