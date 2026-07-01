# This module contains some of the tools used to actually generate
# tests from existing indices.
#
# UBI metric ground-truth (query/document CTR) is computed via the SAME path the
# agent under test uses: raw aggregations are fetched through the OpenSearch MCP
# server (src.utils.mcp_retrieval.search) and handed to the shared compute tools
# (tools.art.ubi_metrics_tools). The search-relevance plugin reads further below
# still use the direct OpenSearch client.
#
# NOTE: ``time_range_days`` applies a last-N-days timestamp filter when set
# (default 30); pass None to compute over all data. Displayed values are rounded
# half-up (see _round2) to match the agent's conventional rounding.
import json
from decimal import ROUND_HALF_UP, Decimal

from src.test_gen.opensearch_client import OpenSearchClientManager, get_client_manager
from src.utils.mcp_retrieval import search
from tools.art.ubi_metrics_tools import compute_document_ctr, compute_ubi_metrics
from utils.logging_helpers import get_logger
from utils.tool_utils import format_tool_error, log_tool_error

logger = get_logger(__name__)

CLICK_ACTION = "click"
IMPRESSION_ACTION = "impression"
ACTION_FIELD = "action_name"
QUERY_TEXT_FIELD = "user_query"
QUERY_ID_FIELD = "query_id"
OBJECT_ID_FIELD = "event_attributes.object.object_id"
POSITION_FIELD = "event_attributes.position.ordinal"
TIMESTAMP_FIELD = "timestamp"

# A separate query table (one row per search) is always assumed. UBI pairs
# ubi_events with ubi_queries by default; any other source must name both tables.
_QUERY_INDEX_BY_EVENTS = {
    "ubi_events": "ubi_queries",
}


def _resolve_query_index(events_index: str, query_index: str | None) -> str:
    """Resolve the query index that holds search volume (one row per search).

    - an explicit ``query_index`` wins;
    - the UBI events index pairs with ``ubi_queries`` by default;
    - any other events index requires an explicit ``query_index``.
    """
    if query_index is not None:
        return query_index
    if events_index in _QUERY_INDEX_BY_EVENTS:
        return _QUERY_INDEX_BY_EVENTS[events_index]
    raise ValueError(
        f"No query index configured for events index '{events_index}'. UBI "
        f"defaults to ubi_queries; for any other source pass query_index "
        f"explicitly — a separate query table is always assumed."
    )


async def list_experiment() -> str:
    """
    Lists all available search relevance experiments.

    Returns:
        str: JSON string containing list of experiments
    """
    try:
        client_manager: OpenSearchClientManager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        response = sr_client.get_experiments()

        return json.dumps(response, indent=2)
    except Exception as e:
        return format_tool_error(f"Error listing experiments: {str(e)}")


async def list_judgment_list() -> str:
    """
    Lists all available judgment lists for search relevance evaluation.

    Returns:
        str: JSON string containing list of judgments
    """
    try:
        client_manager: OpenSearchClientManager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        # Get all judgments
        response = sr_client.get_judgments()

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error listing judgments: {str(e)}")


async def get_judgment(judgment_id: str) -> str:
    """
    Retrieves a specific judgment by ID.

    Args:
        judgment_id: ID of the judgment to retrieve

    Returns:
        str: JSON string containing the judgment details
    """
    try:
        client_manager: OpenSearchClientManager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        response = sr_client.get_judgments(judgment_id=judgment_id)

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error retrieving judgment: {str(e)}")


async def list_query_set() -> str:
    """
    Lists all available query sets.

    Returns:
        str: JSON string containing list of query sets
    """
    try:
        client_manager: OpenSearchClientManager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        response = sr_client.get_query_sets()

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error listing query sets: {str(e)}")


async def get_query_set(query_set_id: str) -> str:
    """
    Retrieves a specific query set by ID.

    Args:
        query_set_id: ID of the query set to retrieve

    Returns:
        str: JSON string containing the query set details
    """
    try:
        client_manager: OpenSearchClientManager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        response = sr_client.get_query_sets(query_set_id=query_set_id)

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error retrieving query set: {str(e)}")


async def list_search_configuration() -> str:
    """
    Lists all available search configurations.

    Returns:
        str: JSON string containing list of search configurations
    """
    try:
        client_manager: OpenSearchClientManager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        # Get all search configurations
        response = sr_client.get_search_configurations()

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error listing search configurations: {str(e)}")


# ---------------------------------------------------------------------------
# UBI metrics — retrieved via MCP SearchIndexTool, computed via the shared
# compute_ubi_metrics / compute_document_ctr tools (the agent's path).
# ---------------------------------------------------------------------------


def _hits_total(resp: dict) -> int:
    return int(((resp.get("hits") or {}).get("total") or {}).get("value", 0))


def _round2(x: float) -> float:
    """Round half-UP to 2 decimals, matching conventional (and LLM) rounding
    rather than Python's round-half-to-even (e.g. 4.625 -> 4.63, not 4.62)."""
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _time_range_filter(time_range_days: int | None) -> dict | None:
    """Range clause for the last ``time_range_days`` days, or None for no filter.

    Uses the same ``now-Nd/d`` .. ``now/d`` date math the agent applies for a
    stated window, so windowed ground-truth aligns with the agent's queries.
    """
    if not time_range_days:
        return None
    return {
        "range": {TIMESTAMP_FIELD: {"gte": f"now-{time_range_days}d/d", "lte": "now/d"}}
    }


def _filtered(must: list[dict], time_range_days: int | None) -> dict:
    """Build a query clause from must-clauses plus an optional time-range filter."""
    time_filter = _time_range_filter(time_range_days)
    if not must and time_filter is None:
        return {"match_all": {}}
    bool_q: dict = {}
    if must:
        bool_q["must"] = must
    if time_filter is not None:
        bool_q["filter"] = [time_filter]
    return {"bool": bool_q}


def _to_click_result(q: dict) -> dict:
    """Map a compute_ubi_metrics per-query entry to the legacy ClickResult shape."""
    return {
        "query_text": q["query"],
        "search_volume": q["search_volume"],
        "searches_with_clicks": q["searches_with_clicks"],
        "total_clicks": q["total_clicks"],
        "ctr_percentage": _round2(q["ctr"] * 100),
        "average_clicks_per_search": q["clicks_per_search"],
        "zero_click_rate_percentage": _round2(q["zero_click_rate"] * 100),
    }


async def _compute_query_breakdown(
    events_index: str,
    query_index: str,
    top_n: int,
    min_search_volume: int = 0,
    time_range_days: int | None = 30,
) -> dict:
    """Fetch the canonical UBI aggregations via MCP and run compute_ubi_metrics.

    A separate query table and event table are always used. Search volume comes
    from ``query_index`` (one row per search), so the ``doc_count`` is the exact
    search volume. Clicks and searches-with-clicks come from ``events_index``;
    searches-with-clicks (distinct clicked query_ids per query) is counted
    exactly by enumerating query_ids (a ``terms`` sub-agg, counted) rather than
    the approximate ``cardinality`` aggregation. Assumes the enumerated counts
    stay within OpenSearch's terms ``size`` / ``search.max_buckets`` limits
    (fine at the synthetic eval-data scale).
    """
    click_must = [{"term": {ACTION_FIELD: CLICK_ACTION}}]

    # Search volume = doc_count of the query table (one row per search) — exact.
    total_queries = _hits_total(
        search(query_index, {"query": _filtered([], time_range_days)})
    )
    svb = search(query_index, {
        "query": _filtered([], time_range_days),
        "aggs": {"by_q": {"terms": {"field": QUERY_TEXT_FIELD, "size": 1000}}},
    })
    sv_buckets = svb["aggregations"]["by_q"]["buckets"]

    total_clicks = _hits_total(
        search(events_index, {"query": _filtered(click_must, time_range_days)})
    )
    qwc = search(events_index, {
        "query": _filtered(click_must, time_range_days),
        "aggs": {"u": {"cardinality": {"field": QUERY_ID_FIELD}}},
    })
    queries_with_clicks = int(qwc["aggregations"]["u"].get("value", 0))

    cb = search(events_index, {
        "query": _filtered(click_must, time_range_days),
        "aggs": {
            "by_q": {
                "terms": {"field": QUERY_TEXT_FIELD, "size": 1000},
                "aggs": {"qids": {"terms": {"field": QUERY_ID_FIELD, "size": 100000}}},
            },
            "missing_query_text": {"missing": {"field": QUERY_TEXT_FIELD}},
        },
    })
    # searches_with_clicks = exact distinct clicked query_ids per query (counted
    # from the enumerated buckets), reshaped to the {"value": n} form that
    # compute_ubi_metrics reads.
    click_buckets = [
        {
            "key": b["key"],
            "doc_count": b["doc_count"],
            "searches_with_clicks": {"value": len((b.get("qids") or {}).get("buckets", []))},
        }
        for b in cb["aggregations"]["by_q"]["buckets"]
    ]
    clicks_without = int(cb["aggregations"]["missing_query_text"].get("doc_count", 0))

    return json.loads(await compute_ubi_metrics(
        total_queries=total_queries,
        total_clicks=total_clicks,
        queries_with_clicks=queries_with_clicks,
        search_volume_buckets=json.dumps(sv_buckets),
        click_buckets=json.dumps(click_buckets),
        clicks_without_query_text=clicks_without,
        top_n=top_n,
        min_search_volume=min_search_volume,
    ))


async def get_query_ctr(
    query_text: str,
    time_range_days: int | None = 30,
    ubi_index: str = "ubi_events",
    query_index: str | None = None,
) -> str:
    """
    Calculate click-through rate for a specific query.

    Args:
        query_text: The search query text to analyze
        time_range_days: When set (e.g. 30), restrict to the last N days; when
            None, compute over all data. Echoed into the result.
        ubi_index: Name of the UBI events index (default: "ubi_events")
        query_index: Index holding search volume. None auto-resolves (ubi_queries
            for ubi_events, otherwise the events index itself in single-index mode).

    Returns:
        str: JSON string with query CTR metrics including total searches,
             searches with clicks, CTR percentage, and average clicks per search
    """
    try:
        query_index = _resolve_query_index(ubi_index, query_index)
        metrics = await _compute_query_breakdown(
            ubi_index, query_index, top_n=100000, time_range_days=time_range_days
        )
        match = next(
            (q for q in metrics.get("top_queries_by_ctr", []) if q["query"] == query_text),
            None,
        )
        if match is None:
            result = {
                "query_text": query_text,
                "time_range_days": time_range_days,
                "total_searches": 0,
                "searches_with_clicks": 0,
                "total_clicks": 0,
                "ctr_percentage": 0,
                "average_clicks_per_search": 0,
                "zero_click_rate_percentage": 0,
            }
        else:
            cr = _to_click_result(match)
            result = {
                "query_text": query_text,
                "time_range_days": time_range_days,
                "total_searches": cr["search_volume"],
                "searches_with_clicks": cr["searches_with_clicks"],
                "total_clicks": cr["total_clicks"],
                "ctr_percentage": cr["ctr_percentage"],
                "average_clicks_per_search": cr["average_clicks_per_search"],
                "zero_click_rate_percentage": cr["zero_click_rate_percentage"],
            }
        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(logger, f"Error calculating query CTR: {str(e)}")


async def get_document_ctr(
    doc_id: str, time_range_days: int | None = 30, ubi_index: str = "ubi_events"
) -> str:
    """
    Calculate click-through rate for a specific document.

    Args:
        doc_id: The document ID to analyze
        time_range_days: When set (e.g. 30), restrict to the last N days; when
            None, compute over all data. Echoed into the result.
        ubi_index: Name of the UBI events index (default: "ubi_events")

    Returns:
        str: JSON string with document CTR metrics including total impressions,
             total clicks, CTR percentage, and average position when clicked
    """
    try:
        impressions = _hits_total(search(ubi_index, {
            "query": _filtered(
                [
                    {"term": {OBJECT_ID_FIELD: doc_id}},
                    {"term": {ACTION_FIELD: IMPRESSION_ACTION}},
                ],
                time_range_days,
            ),
        }))
        clk = search(ubi_index, {
            "query": _filtered(
                [
                    {"term": {OBJECT_ID_FIELD: doc_id}},
                    {"term": {ACTION_FIELD: CLICK_ACTION}},
                ],
                time_range_days,
            ),
            "aggs": {"avg_position": {"avg": {"field": POSITION_FIELD}}},
        })
        clicks = _hits_total(clk)
        avg_position = clk["aggregations"]["avg_position"].get("value")

        out = json.loads(await compute_document_ctr(
            document_id=doc_id,
            impressions=impressions,
            clicks=clicks,
            avg_click_position=avg_position,
        ))
        result = {
            "document_id": doc_id,
            "time_range_days": time_range_days,
            "total_impressions": out["impressions"],
            "total_clicks": out["clicks"],
            "ctr_percentage": _round2(out["ctr"] * 100),
            "average_position_when_clicked": _round2(out["avg_click_position"])
            if out["avg_click_position"] is not None
            else None,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(logger, f"Error calculating document CTR: {str(e)}")


async def get_query_performance_metrics(
    query_text: str | None = None,
    top_n: int = 20,
    time_range_days: int | None = 30,
    ubi_index: str = "ubi_events",
    query_index: str | None = None,
) -> str:
    """
    Get comprehensive performance metrics for queries.
    If query_text provided: detailed metrics for that query.
    If query_text is None: top N queries (by CTR) with their metrics.

    Args:
        query_text: Specific query to analyze (optional)
        top_n: Number of top queries to return if query_text not provided
        time_range_days: When set (e.g. 30), restrict to the last N days; when
            None, compute over all data. Echoed into the result.
        ubi_index: Name of the UBI events index (default: "ubi_events")
        query_index: Index holding search volume. None auto-resolves (ubi_queries
            for ubi_events, otherwise the events index itself in single-index mode).

    Returns:
        str: JSON string with performance metrics for query/queries
    """
    try:
        if query_text:
            return await get_query_ctr(query_text, time_range_days, ubi_index, query_index)

        query_index = _resolve_query_index(ubi_index, query_index)
        metrics = await _compute_query_breakdown(
            ubi_index, query_index, top_n=top_n, time_range_days=time_range_days
        )
        queries = [_to_click_result(q) for q in metrics.get("top_queries_by_ctr", [])]
        result = {
            "time_range_days": time_range_days,
            "total_queries_analyzed": len(queries),
            "queries": queries,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(
            logger, f"Error getting query performance metrics: {str(e)}"
        )


async def get_top_queries_by_engagement(
    top_n: int = 20,
    min_search_volume: int = 5,
    time_range_days: int | None = 30,
    ubi_index: str = "ubi_events",
    query_index: str | None = None,
) -> str:
    """
    Get queries with highest CTR (best engagement).
    Filters out low-volume queries to ensure statistical significance.

    Args:
        top_n: Number of top queries to return (default: 20)
        min_search_volume: Minimum number of searches required (default: 5)
        time_range_days: When set (e.g. 30), restrict to the last N days; when
            None, compute over all data. Echoed into the result.
        ubi_index: Name of the UBI events index (default: "ubi_events")
        query_index: Index holding search volume. None auto-resolves (ubi_queries
            for ubi_events, otherwise the events index itself in single-index mode).

    Returns:
        str: JSON string with top queries by CTR
    """
    try:
        query_index = _resolve_query_index(ubi_index, query_index)
        metrics = await _compute_query_breakdown(
            ubi_index,
            query_index,
            top_n=top_n,
            min_search_volume=min_search_volume,
            time_range_days=time_range_days,
        )
        queries = [_to_click_result(q) for q in metrics.get("top_queries_by_ctr", [])]
        result = {
            "time_range_days": time_range_days,
            "min_search_volume": min_search_volume,
            "total_queries_analyzed": len(queries),
            "queries": queries,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(
            logger, f"Error getting top queries by engagement: {str(e)}"
        )


async def get_top_documents_by_engagement(
    top_n: int = 20,
    min_impressions: int = 5,
    time_range_days: int | None = 30,
    ubi_index: str = "ubi_events",
) -> str:
    """
    Get documents with highest CTR (best engagement).
    Filters out low-impression documents to ensure statistical significance.

    Args:
        top_n: Number of top documents to return (default: 20)
        min_impressions: Minimum number of impressions required (default: 5)
        time_range_days: When set (e.g. 30), restrict to the last N days; when
            None, compute over all data. Echoed into the result.
        ubi_index: Name of the UBI events index (default: "ubi_events")

    Returns:
        str: JSON string with top documents by CTR
    """
    try:
        resp = search(ubi_index, {
            "query": _filtered([], time_range_days),
            "aggs": {
                "by_doc": {
                    "terms": {"field": OBJECT_ID_FIELD, "size": 100},
                    "aggs": {
                        "impressions": {"filter": {"term": {ACTION_FIELD: IMPRESSION_ACTION}}},
                        "clicks": {
                            "filter": {"term": {ACTION_FIELD: CLICK_ACTION}},
                            "aggs": {"avg_position": {"avg": {"field": POSITION_FIELD}}},
                        },
                    },
                },
            },
        })
        buckets = resp["aggregations"]["by_doc"]["buckets"]

        out = json.loads(await compute_document_ctr(
            document_buckets=json.dumps(buckets),
            min_impressions=min_impressions,
            top_n=top_n,
        ))
        documents = [
            {
                "document_id": d["document_id"],
                "total_impressions": d["impressions"],
                "total_clicks": d["clicks"],
                "ctr_percentage": _round2(d["ctr"] * 100),
                "average_position_when_clicked": _round2(d["avg_click_position"])
                if d["avg_click_position"] is not None
                else None,
            }
            for d in out.get("top_documents_by_ctr", [])
        ]
        result = {
            "time_range_days": time_range_days,
            "min_impressions": min_impressions,
            "total_documents_analyzed": len(documents),
            "documents": documents,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(
            logger, f"Error getting top documents by engagement: {str(e)}"
        )
