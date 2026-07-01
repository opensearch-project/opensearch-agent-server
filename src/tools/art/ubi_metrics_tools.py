"""
UBI Metrics Tools

Provides deterministic computation of user behavior metrics from data
retrieved via SearchIndexTool (OpenSearch MCP). The LLM retrieves raw
aggregation results; these tools do all arithmetic to prevent hallucinated
calculations.

Metric definitions (read carefully — these are the canonical, single-source
definitions used both by the agent and by eval test generation):

- **Query CTR** (``ctr`` / ``ctr_pct``): the fraction of *searches* that
  received at least one click — ``searches_with_clicks / searches``. This is a
  true rate, always in [0, 1]. Overall it needs ``queries_with_clicks``;
  per-query it is derived from the click buckets.
- **clicks_per_search**: the average number of clicks per search —
  ``total_clicks / searches``. This is an engagement-depth metric and is NOT a
  rate; it can exceed 1 (one search may yield several clicks). It must never be
  called CTR.
- **zero_click_rate**: ``1 - ctr`` — the fraction of searches with no click.
- **Document CTR** (``compute_document_ctr``): ``clicks / impressions`` where
  impressions are real ``action_name:impression`` events for the document.
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from utils.logging_helpers import get_logger, log_info_event
from utils.monitored_tool import monitored_tool

logger = get_logger(__name__)


def _pct(fraction: float) -> str:
    """Format a [0, 1] fraction as a two-decimal percentage string."""
    return f"{fraction * 100:.2f}%"


def _round2(x: float) -> float:
    """Round half-UP to 2 decimals (conventional rounding, e.g. 4.625 -> 4.63),
    so displayed values match how the LLM/humans round rather than Python's
    round-half-to-even."""
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


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
        "Computes query/search-level user behavior metrics from pre-aggregated "
        "OpenSearch counts. "
        "ALWAYS use this tool instead of computing metrics yourself — arithmetic must "
        "not be delegated to the LLM. "
        "CTR is the fraction of searches with at least one click (0-100%); pass "
        "total_queries and queries_with_clicks to get it. total_clicks yields "
        "clicks_per_search (clicks per search, can exceed 1 — NOT a CTR). "
        "Pass search_volume_buckets and click_buckets (terms on the query-text "
        "field) for a per-query breakdown."
    ),
)
def compute_ubi_metrics(
    total_queries: int,
    total_clicks: int,
    queries_with_clicks: int | None = None,
    search_volume_buckets: str | None = None,
    click_buckets: str | None = None,
    searches_with_clicks_agg: str = "searches_with_clicks",
    clicks_without_query_text: int | None = None,
    top_n: int = 10,
    min_search_volume: int = 0,
) -> str:
    """Compute query/search-level UBI metrics from pre-aggregated OpenSearch data.

    Accepts raw aggregation bucket lists directly from SearchIndexTool responses
    so that all arithmetic (division, summing across query_ids, ranking) is done
    in Python rather than by the LLM.

    Args:
        total_queries: Total document count from the ubi_queries index
            (``hits.total.value`` from a match_all query with size=0). This is
            the number of searches.
        total_clicks: Total document count from ubi_events filtered to the
            click action (``hits.total.value`` with size=0 and action filter).
        queries_with_clicks: Cardinality of unique query_ids in ubi_events
            that received at least one click. REQUIRED to compute the overall
            CTR and zero-click rate. Obtain via a cardinality aggregation on
            query_id in ubi_events filtered to the click action.
        search_volume_buckets: JSON array of terms-aggregation buckets from
            ubi_queries grouped by the query-text field (use the field name
            directly when it is keyword-typed — no ``.keyword`` suffix). Each
            bucket must have ``"key"`` (query text) and ``"doc_count"`` (the
            search volume for that query). Pass the
            ``aggregations.<agg_name>.buckets`` array directly.
            Example: ``'[{"key": "laptop", "doc_count": 120}, ...]'``
        click_buckets: JSON array of terms-aggregation buckets from ubi_events
            (filtered to the click action) grouped by the same query-text field,
            each with a cardinality sub-aggregation on query_id named
            ``<searches_with_clicks_agg>``. Each bucket must have ``"key"``
            (query text), ``"doc_count"`` (total clicks for that query) and the
            cardinality sub-aggregation giving the number of distinct searches
            (query_ids) that received a click. Click events whose query-text
            field is null are naturally excluded by this aggregation — pass
            their count via ``clicks_without_query_text`` for transparency.
            Pass the ``aggregations.<agg_name>.buckets`` array directly.
            Example: ``'[{"key": "laptop", "doc_count": 30,
            "searches_with_clicks": {"value": 12}}, ...]'``
        searches_with_clicks_agg: Name of the cardinality(query_id) sub-aggregation
            inside each click bucket. Defaults to ``"searches_with_clicks"``.
        clicks_without_query_text: Optional count of click events with no query
            text (the ``missing`` bucket), echoed into the output so the
            unattributed clicks are visible.
        top_n: Number of entries to include in the ranked per-query lists.
        min_search_volume: Drop queries with fewer than this many searches from
            the per-query ranking (statistical-significance filter). Does not
            affect ``per_query_count``.

    Returns:
        JSON string with computed metrics: overall CTR and zero-click rate (when
        ``queries_with_clicks`` is provided), clicks-per-search, and a per-query
        breakdown ranked by CTR (when both bucket arguments are provided).
    """
    results: dict[str, Any] = {
        "total_queries": total_queries,
        "total_clicks": total_clicks,
    }

    # --- Clicks per search (engagement depth; NOT a rate, may exceed 1) ---
    if total_queries > 0:
        results["clicks_per_search"] = _round2(total_clicks / total_queries)
    else:
        results["clicks_per_search"] = 0.0
        results["note"] = "No queries recorded — metrics are undefined."

    # --- Overall CTR + zero-click rate (bounded; require queries_with_clicks) ---
    if queries_with_clicks is not None and total_queries > 0:
        ctr = queries_with_clicks / total_queries
        zero_click_rate = 1 - ctr
        results["queries_with_clicks"] = queries_with_clicks
        results["queries_without_clicks"] = total_queries - queries_with_clicks
        results["ctr"] = round(ctr, 4)
        results["ctr_pct"] = _pct(ctr)
        results["zero_click_rate"] = round(zero_click_rate, 4)
        results["zero_click_rate_pct"] = _pct(zero_click_rate)
    elif queries_with_clicks is None:
        results["ctr_note"] = (
            "Provide queries_with_clicks to compute CTR (fraction of searches "
            "with a click)."
        )

    # --- Per-query breakdown ---
    if search_volume_buckets is not None and click_buckets is not None:
        try:
            sv_buckets = _parse_buckets(search_volume_buckets, "search_volume_buckets")
            clk_buckets = _parse_buckets(click_buckets, "click_buckets")
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        # search volume keyed by query text (terms on the query-text field over
        # ubi_queries)
        search_volume_by_query: dict[str, int] = {
            b["key"]: int(b["doc_count"]) for b in sv_buckets if "key" in b
        }

        # clicks and searches-with-clicks keyed by query text (terms on the
        # query-text field over click events, with a cardinality(query_id)
        # sub-aggregation). Null-query-text clicks never appear here.
        clicks_by_query: dict[str, int] = {}
        searches_with_clicks_by_query: dict[str, int] = {}
        for bucket in clk_buckets:
            query_text = bucket.get("key")
            if query_text is None:
                continue
            clicks_by_query[query_text] = int(bucket.get("doc_count", 0))
            searches_with_clicks_by_query[query_text] = int(
                (bucket.get(searches_with_clicks_agg) or {}).get("value", 0)
            )

        if clicks_without_query_text:
            results["clicks_without_query_text"] = clicks_without_query_text

        # join and compute per-query metrics
        all_query_texts = search_volume_by_query.keys() | clicks_by_query.keys()
        per_query: list[dict[str, Any]] = []
        for qt in all_query_texts:
            search_volume = search_volume_by_query.get(qt, 0)
            clicks = clicks_by_query.get(qt, 0)
            searches_with_clicks = searches_with_clicks_by_query.get(qt, 0)
            if search_volume > 0:
                ctr = searches_with_clicks / search_volume
                clicks_per_search = clicks / search_volume
                zero_click_rate = 1 - ctr
            else:
                ctr = clicks_per_search = zero_click_rate = 0.0
            per_query.append({
                "query": qt,
                "search_volume": search_volume,
                "searches_with_clicks": searches_with_clicks,
                "total_clicks": clicks,
                "ctr": round(ctr, 4),
                "ctr_pct": _pct(ctr),
                "clicks_per_search": _round2(clicks_per_search),
                "zero_click_rate": round(zero_click_rate, 4),
                "zero_click_rate_pct": _pct(zero_click_rate),
            })

        ranked = [q for q in per_query if q["search_volume"] >= min_search_volume]
        ranked.sort(key=lambda x: x["ctr"], reverse=True)
        results["per_query_count"] = len(per_query)
        results["ranked_query_count"] = len(ranked)
        results["top_queries_by_ctr"] = ranked[:top_n]
        results["bottom_queries_by_ctr"] = ranked[-top_n:]

    log_info_event(
        logger,
        "UBI metrics computed",
        "ubi_metrics.computed",
        total_queries=total_queries,
        total_clicks=total_clicks,
        has_per_query="per_query_count" in results,
    )

    return json.dumps(results, indent=2)


@monitored_tool(
    name="ComputeDocumentCTRTool",
    description=(
        "Computes document-level CTR (clicks / impressions) from pre-aggregated "
        "OpenSearch counts, where impressions are real impression events. "
        "ALWAYS use this tool instead of computing CTR yourself. "
        "Pass document_id + impressions + clicks (+ optional avg_click_position) "
        "for a single document, OR document_buckets for a ranked multi-document "
        "breakdown."
    ),
)
def compute_document_ctr(
    document_id: str | None = None,
    impressions: int | None = None,
    clicks: int | None = None,
    avg_click_position: float | None = None,
    document_buckets: str | None = None,
    impressions_agg: str = "impressions",
    clicks_agg: str = "clicks",
    position_agg: str = "avg_position",
    min_impressions: int = 0,
    top_n: int = 10,
) -> str:
    """Compute document-level CTR from pre-aggregated OpenSearch data.

    Two modes:

    - **Single document**: pass ``document_id``, ``impressions``, ``clicks`` and
      optionally ``avg_click_position``. Returns one object.
    - **Multiple documents**: pass ``document_buckets`` — a terms-aggregation on
      the document-id field where each bucket carries impression-count and
      click-count filter sub-aggregations and (optionally) an average click
      position. Returns documents ranked by CTR.

    Args:
        document_id: Document id (single-document mode).
        impressions: Impression-event count for the document (single mode).
        clicks: Click-event count for the document (single mode).
        avg_click_position: Average result position at click time (single mode).
        document_buckets: JSON array of terms-aggregation buckets grouped by the
            document-id field. Each bucket must have ``"key"`` (document id), an
            impressions filter sub-agg named ``<impressions_agg>`` with a
            ``"doc_count"``, and a clicks filter sub-agg named ``<clicks_agg>``
            with a ``"doc_count"`` and (optionally) an average-position sub-agg
            named ``<position_agg>`` with a ``"value"``.
            Pass the ``aggregations.<agg_name>.buckets`` array directly.
        impressions_agg: Name of the impressions filter sub-aggregation.
        clicks_agg: Name of the clicks filter sub-aggregation.
        position_agg: Name of the average-position sub-aggregation inside the
            clicks sub-aggregation.
        min_impressions: Drop documents with fewer than this many impressions
            from the ranking (statistical-significance filter).
        top_n: Number of entries to include in the ranked lists.

    Returns:
        JSON string with the computed document CTR(s).
    """
    def _doc_metrics(
        doc_id: str | None, imp: int, clk: int, position: float | None
    ) -> dict[str, Any]:
        ctr = clk / imp if imp > 0 else 0.0
        return {
            "document_id": doc_id,
            "impressions": imp,
            "clicks": clk,
            "ctr": round(ctr, 4),
            "ctr_pct": _pct(ctr),
            "avg_click_position": _round2(position) if position is not None else None,
        }

    # --- Single-document mode ---
    if document_buckets is None:
        if impressions is None or clicks is None:
            return json.dumps({
                "error": (
                    "Single-document mode requires impressions and clicks; "
                    "or pass document_buckets for a multi-document breakdown."
                )
            })
        result = _doc_metrics(document_id, impressions, clicks, avg_click_position)
        log_info_event(
            logger,
            "Document CTR computed",
            "ubi_metrics.document_ctr_computed",
            document_id=document_id,
            mode="single",
        )
        return json.dumps(result, indent=2)

    # --- Multi-document mode ---
    try:
        buckets = _parse_buckets(document_buckets, "document_buckets")
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    documents: list[dict[str, Any]] = []
    for bucket in buckets:
        imp = int(bucket.get(impressions_agg, {}).get("doc_count", 0))
        if imp < min_impressions:
            continue
        clk_sub = bucket.get(clicks_agg, {})
        clk = int(clk_sub.get("doc_count", 0))
        position = clk_sub.get(position_agg, {}).get("value")
        documents.append(_doc_metrics(bucket.get("key"), imp, clk, position))

    documents.sort(key=lambda x: x["ctr"], reverse=True)

    results = {
        "min_impressions": min_impressions,
        "document_count": len(documents),
        "top_documents_by_ctr": documents[:top_n],
        "bottom_documents_by_ctr": documents[-top_n:],
    }
    log_info_event(
        logger,
        "Document CTR computed",
        "ubi_metrics.document_ctr_computed",
        document_count=len(documents),
        mode="multi",
    )
    return json.dumps(results, indent=2)
