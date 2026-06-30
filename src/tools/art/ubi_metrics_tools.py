"""
UBI Metrics Tools

Provides deterministic computation of user behavior metrics from data
retrieved via SearchIndexTool (OpenSearch MCP). The LLM retrieves raw
aggregation results; this tool does all arithmetic to prevent hallucinated
calculations.
"""

from __future__ import annotations

import json
from typing import Any

from utils.logging_helpers import get_logger, log_info_event
from utils.monitored_tool import monitored_tool

logger = get_logger(__name__)


def _parse_buckets(raw: str, label: str) -> list[dict[str, Any]]:
    """Parse a JSON bucket list, raising ValueError with a clear message on failure."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {label}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"{label} must be a JSON array of bucket objects, got {type(data).__name__}")
    return data


@monitored_tool(
    name="ComputeUBIMetricsTool",
    description=(
        "Computes user behavior metrics (CTR, zero-click rate, per-query CTR ranking) "
        "from pre-aggregated OpenSearch counts. "
        "ALWAYS use this tool instead of computing metrics yourself — arithmetic must "
        "not be delegated to the LLM. "
        "Pass total_queries and total_clicks for global CTR. "
        "Pass impression_buckets and click_query_id_buckets for a per-query breakdown."
    ),
)
def compute_ubi_metrics(
    total_queries: int,
    total_clicks: int,
    queries_with_clicks: int | None = None,
    impression_buckets: str | None = None,
    click_query_id_buckets: str | None = None,
    query_text_field: str = "user_query",
    click_query_text_agg: str = "query_text",
) -> str:
    """Compute UBI metrics from pre-aggregated OpenSearch data.

    Accepts raw aggregation bucket lists directly from SearchIndexTool responses
    so that all arithmetic (division, summing across query_ids, ranking) is done
    in Python rather than by the LLM.

    Args:
        total_queries: Total document count from the ubi_queries index
            (``hits.total.value`` from a match_all query with size=0).
        total_clicks: Total document count from ubi_events filtered to the
            click action (``hits.total.value`` with size=0 and action filter).
        queries_with_clicks: Cardinality of unique query_ids in ubi_events
            that received at least one click. Used to compute zero-click rate.
            Obtain via a cardinality aggregation on query_id in ubi_events
            filtered to the click action.
        impression_buckets: JSON array of terms-aggregation buckets from
            ubi_queries grouped by the query-text field (e.g. user_query.keyword).
            Each bucket must have ``"key"`` (query text) and ``"doc_count"``
            (impression count). Pass the ``aggregations.<agg_name>.buckets`` array
            directly from the SearchIndexTool response.
            Example: ``'[{"key": "laptop", "doc_count": 120}, ...]'``
        click_query_id_buckets: JSON array of terms-aggregation buckets from
            ubi_events (filtered to the click action) grouped by query_id, with
            a top_hits sub-aggregation named ``<click_query_text_agg>`` that
            fetches one source document to recover the query text.
            Each bucket must have ``"key"`` (query_id), ``"doc_count"``
            (click count), and the top_hits sub-aggregation keyed by
            ``<click_query_text_agg>``.
            Pass the ``aggregations.<agg_name>.buckets`` array directly.
        query_text_field: Field name in the top_hits ``_source`` that holds
            the query text. Defaults to ``"user_query"``.
        click_query_text_agg: Name of the top_hits sub-aggregation inside each
            click bucket that provides the query text. Defaults to
            ``"query_text"``. Must match the name used in the aggregation query.

    Returns:
        JSON string with computed metrics: overall CTR, zero-click rate (when
        ``queries_with_clicks`` is provided), and per-query CTR ranking (when
        both bucket arguments are provided).
    """
    results: dict[str, Any] = {
        "total_queries": total_queries,
        "total_clicks": total_clicks,
    }

    # --- Global CTR ---
    if total_queries > 0:
        ctr = total_clicks / total_queries
        results["overall_ctr"] = round(ctr, 4)
        results["overall_ctr_pct"] = f"{ctr * 100:.2f}%"
    else:
        results["overall_ctr"] = 0.0
        results["overall_ctr_pct"] = "0.00%"
        results["note"] = "No queries recorded — CTR is undefined."

    # --- Zero-click rate ---
    if queries_with_clicks is not None and total_queries > 0:
        zero_click = total_queries - queries_with_clicks
        zero_click_rate = zero_click / total_queries
        results["queries_with_clicks"] = queries_with_clicks
        results["queries_without_clicks"] = zero_click
        results["zero_click_rate"] = round(zero_click_rate, 4)
        results["zero_click_rate_pct"] = f"{zero_click_rate * 100:.2f}%"

    # --- Per-query CTR ---
    if impression_buckets is not None and click_query_id_buckets is not None:
        try:
            imp_buckets = _parse_buckets(impression_buckets, "impression_buckets")
            clk_buckets = _parse_buckets(click_query_id_buckets, "click_query_id_buckets")
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        # impression counts keyed by query text
        impressions_by_query: dict[str, int] = {
            b["key"]: int(b["doc_count"]) for b in imp_buckets if "key" in b
        }

        # sum click counts by query text, recovering text from top_hits sub-agg
        clicks_by_query: dict[str, int] = {}
        skipped = 0
        for bucket in clk_buckets:
            click_count = int(bucket.get("doc_count", 0))
            top_hits_agg = bucket.get(click_query_text_agg)
            if not top_hits_agg:
                skipped += 1
                continue
            hits = top_hits_agg.get("hits", {}).get("hits", [])
            if not hits:
                skipped += 1
                continue
            query_text = hits[0].get("_source", {}).get(query_text_field)
            if not query_text:
                skipped += 1
                continue
            clicks_by_query[query_text] = clicks_by_query.get(query_text, 0) + click_count

        if skipped:
            results["skipped_click_buckets"] = skipped
            results["skipped_reason"] = (
                f"Could not recover query text from '{click_query_text_agg}' top_hits "
                f"sub-aggregation or '{query_text_field}' field for {skipped} bucket(s)."
            )

        # join and compute per-query CTR
        all_query_texts = impressions_by_query.keys() | clicks_by_query.keys()
        per_query: list[dict[str, Any]] = []
        for qt in all_query_texts:
            impressions = impressions_by_query.get(qt, 0)
            clicks = clicks_by_query.get(qt, 0)
            ctr = clicks / impressions if impressions > 0 else 0.0
            per_query.append({
                "query": qt,
                "impressions": impressions,
                "clicks": clicks,
                "ctr": round(ctr, 4),
                "ctr_pct": f"{ctr * 100:.2f}%",
            })

        per_query.sort(key=lambda x: x["ctr"], reverse=True)
        results["per_query_count"] = len(per_query)
        results["top_queries_by_ctr"] = per_query[:10]
        results["bottom_queries_by_ctr"] = per_query[-10:]

    log_info_event(
        logger,
        "UBI metrics computed",
        "ubi_metrics.computed",
        total_queries=total_queries,
        total_clicks=total_clicks,
        has_per_query="per_query_count" in results,
    )

    return json.dumps(results, indent=2)
