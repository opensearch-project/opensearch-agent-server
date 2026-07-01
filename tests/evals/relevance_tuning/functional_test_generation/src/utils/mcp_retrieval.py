"""MCP-backed retrieval for test generation.

Replaces the direct OpenSearch client for UBI metric ground-truth: it fetches
raw aggregations through the OpenSearch MCP server's ``SearchIndexTool`` so that
test generation uses the *same* data path as the agent under test (the raw
counts are then handed to the same ``compute_ubi_metrics`` /
``compute_document_ctr`` tools the agent uses).

Auth: a ``Basic`` token is injected via ``OboAuth`` (the scheme is forwarded
verbatim to the MCP server, which passes it to OpenSearch).
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from typing import Any

import httpx
from mcp.client.streamable_http import streamable_http_client
from strands.tools.mcp import MCPClient

from utils.logging_helpers import get_logger, log_info_event
from utils.obo_context import OboAuth

logger = get_logger(__name__)

DEFAULT_MCP_SERVER_URL = "http://localhost:3001/mcp"
SEARCH_TOOL = "SearchIndexTool"

_client: MCPClient | None = None
_http_client: httpx.AsyncClient | None = None


def init_mcp(
    mcp_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> None:
    """Initialise the shared MCP client (idempotent)."""
    global _client, _http_client
    if _client is not None:
        return

    mcp_url = mcp_url or os.getenv("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL)
    username = username or os.environ["TEST_GEN_OPENSEARCH_USERNAME"]
    password = password or os.environ["TEST_GEN_OPENSEARCH_PASSWORD"]

    auth = OboAuth()
    auth.set_token("Basic " + base64.b64encode(f"{username}:{password}".encode()).decode())
    _http_client = httpx.AsyncClient(
        auth=auth,
        timeout=httpx.Timeout(30, read=300),
        verify=False,
        follow_redirects=True,
    )
    _client = MCPClient(lambda: streamable_http_client(mcp_url, http_client=_http_client))
    _client.start()
    log_info_event(
        logger,
        f"[test-gen] MCP retrieval initialised ({mcp_url})",
        "test_gen.mcp_init",
        mcp_url=mcp_url,
    )


def _require_client() -> MCPClient:
    if _client is None:
        init_mcp()
    assert _client is not None
    return _client


def search(index: str, query_dsl: dict[str, Any], size: int = 0) -> dict[str, Any]:
    """Run a search via SearchIndexTool and return the parsed OpenSearch response.

    ``query_dsl`` is a full search body (it may include ``query`` and ``aggs``).
    The tool wraps its JSON in a human-readable prefix, which is stripped here.
    """
    client = _require_client()
    res = client.call_tool_sync(
        str(uuid.uuid4()), SEARCH_TOOL,
        {"index": index, "query_dsl": query_dsl, "size": size},
    )
    content = res.content if hasattr(res, "content") else res.get("content", [])
    parts = []
    for block in content:
        txt = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
        if txt:
            parts.append(txt)
    raw = "\n".join(parts)
    brace = raw.find("{")
    if brace < 0:
        raise RuntimeError(f"Unexpected {SEARCH_TOOL} response for '{index}': {raw[:300]}")
    return json.loads(raw[brace:])


def shutdown() -> None:
    """Stop the shared MCP client (best effort)."""
    global _client, _http_client
    if _client is not None:
        try:
            _client.stop(None, None, None)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        _client = None
        _http_client = None
