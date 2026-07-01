"""
Overview Tools

Self-contained, deterministic overviews of search-relevance resources (judgment
lists, query sets). Each tool does the WHOLE job in one call the agent makes:

  1. retrieve the resources through the OpenSearch MCP server (via the shared
     MCP client — no direct source-system access),
  2. sort by recency and keep the last N,
  3. trim each item's long inner list (ratings / queries) to a small sample,
  4. return a compact JSON the agent reports verbatim.

Folding retrieve+truncate into one tool removes the LLM orchestration between
steps (which was causing fabricated or failed judgment overviews) and makes the
sample deterministically match the eval ground truth.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from utils.logging_helpers import get_logger, log_info_event
from utils.monitored_tool import monitored_tool

logger = get_logger(__name__)

# Shared, already-started MCPClient (injected at agent init via set_mcp_client).
_mcp_client: Any = None

# Query body that returns resources newest-first WITH their full _source
# (SearchJudgmentsTool omits the ratings unless _source is requested).
def _recent_query_body(size: int) -> dict[str, Any]:
    return {
        "query": {"match_all": {}},
        "sort": [{"timestamp": {"order": "desc"}}],
        "size": max(size, 1),
        "_source": {"includes": ["*"]},
    }


def set_mcp_client(mcp_client: Any) -> None:
    """Inject the shared, started MCPClient so the overview tools can retrieve."""
    global _mcp_client
    _mcp_client = mcp_client
    log_info_event(
        logger, "[overview] MCP client configured", "overview.mcp_client_configured"
    )


def _mcp_search(tool_name: str, query_body: dict[str, Any]) -> list[dict[str, Any]]:
    """Call an MCP search tool and return the list of hit ``_source`` dicts."""
    if _mcp_client is None:
        raise RuntimeError("MCP client not configured for overview tools")
    res = _mcp_client.call_tool_sync(
        str(uuid.uuid4()), tool_name, {"query_body": query_body}
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
        raise RuntimeError(f"Unexpected {tool_name} response: {raw[:200]}")
    data = json.loads(raw[brace:])
    hits = (data.get("hits") or {}).get("hits") or []
    return [h.get("_source", h) for h in hits]


def _sort_recent(items: list[dict], last_n: int) -> list[dict]:
    return sorted(items, key=lambda x: x.get("timestamp") or "", reverse=True)[:last_n]


@monitored_tool(
    name="GetJudgmentListsOverviewTool",
    description=(
        "Returns a compact overview of the most recent judgment lists — retrieves "
        "them from OpenSearch and truncates in one step. ALWAYS use this for a "
        "judgment-list overview; do NOT call SearchJudgmentsTool yourself. Keeps the "
        "last_n most recent lists and, per list, the first max_queries judged queries "
        "with the first max_ratings docId->rating pairs each. Report EXACTLY what it "
        "returns (ids, timestamps, names, statuses, types, queries, docIds, ratings) — "
        "never add, drop, reorder, or invent anything."
    ),
)
def get_judgment_lists_overview(
    last_n: int = 4,
    max_queries: int = 5,
    max_ratings: int = 5,
) -> str:
    """Compact, truncated overview of the last_n most recent judgment lists."""
    try:
        sources = _mcp_search("SearchJudgmentsTool", _recent_query_body(last_n))
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the agent
        return json.dumps({"error": f"Could not retrieve judgment lists: {exc}"})

    lists: list[dict[str, Any]] = []
    for s in _sort_recent(sources, last_n):
        ratings_out = []
        for jr in (s.get("judgmentRatings") or [])[:max_queries]:
            ratings_out.append({
                "query": jr.get("query"),
                "ratings": [
                    {"docId": r.get("docId"), "rating": r.get("rating")}
                    for r in (jr.get("ratings") or [])[:max_ratings]
                ],
            })
        lists.append({
            "id": s.get("id"),
            "timestamp": s.get("timestamp"),
            "name": s.get("name"),
            "status": s.get("status"),
            "type": s.get("type"),
            "judgmentRatings": ratings_out,
        })
    log_info_event(
        logger, "judgment overview built", "overview.judgments", count=len(lists)
    )
    return json.dumps({"judgment_lists": lists}, indent=2)


@monitored_tool(
    name="GetQuerySetsOverviewTool",
    description=(
        "Returns a compact overview of the most recent query sets — retrieves them "
        "from OpenSearch and truncates in one step. ALWAYS use this for a query-set "
        "overview; do NOT call SearchQuerySetsTool yourself. Keeps the last_n most "
        "recent query sets and, per set, the first max_queries query texts. Report "
        "EXACTLY what it returns (ids, names, descriptions, timestamps, query texts) — "
        "never add, drop, reorder, or invent anything."
    ),
)
def get_query_sets_overview(
    last_n: int = 5,
    max_queries: int = 10,
) -> str:
    """Compact, truncated overview of the last_n most recent query sets."""
    try:
        sources = _mcp_search("SearchQuerySetsTool", _recent_query_body(last_n))
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the agent
        return json.dumps({"error": f"Could not retrieve query sets: {exc}"})

    sets: list[dict[str, Any]] = []
    for s in _sort_recent(sources, last_n):
        queries = [
            q.get("queryText") if isinstance(q, dict) else q
            for q in (s.get("querySetQueries") or [])[:max_queries]
        ]
        sets.append({
            "id": s.get("id"),
            "name": s.get("name"),
            "description": s.get("description"),
            "timestamp": s.get("timestamp"),
            "querySetQueries": queries,
        })
    log_info_event(logger, "query-set overview built", "overview.query_sets", count=len(sets))
    return json.dumps({"query_sets": sets}, indent=2)
