"""
Unit tests for compute_ubi_metrics and compute_document_ctr.

Covers:
- Bounded overall CTR (searches with a click / searches) — requires queries_with_clicks
- clicks_per_search (clicks / searches) — the unbounded engagement metric, never a CTR
- Zero-click rate as 1 - CTR
- Per-query breakdown (bounded CTR, searches_with_clicks, clicks_per_search) from raw buckets
- Queries appearing only in search-volume buckets or only in click buckets
- Click bucket missing its cardinality sub-agg defaults searches_with_clicks to 0
- Invalid JSON inputs return an error key rather than raising
- Document CTR in single- and multi-document modes
"""

import json

import pytest

from tools.art.ubi_metrics_tools import compute_document_ctr, compute_ubi_metrics

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _result(coro) -> dict:
    return json.loads(await coro)


def _sv_buckets(pairs: list[tuple[str, int]]) -> str:
    """Build a search-volume bucket list: [(query_text, doc_count), ...]."""
    return json.dumps([{"key": q, "doc_count": c} for q, c in pairs])


def _click_buckets(
    rows: list[tuple[str, int, int]],
    swc_agg: str = "searches_with_clicks",
) -> str:
    """Build a click bucket list: [(query_text, clicks, searches_with_clicks), ...]."""
    return json.dumps([
        {"key": qt, "doc_count": clicks, swc_agg: {"value": swc}}
        for qt, clicks, swc in rows
    ])


def _doc_buckets(
    rows: list[tuple[str, int, int, float | None]],
    impressions_agg: str = "impressions",
    clicks_agg: str = "clicks",
    position_agg: str = "avg_position",
) -> str:
    """Build a document terms-agg bucket list: [(doc_id, impressions, clicks, avg_pos), ...]."""
    buckets = []
    for doc_id, imp, clk, pos in rows:
        buckets.append({
            "key": doc_id,
            impressions_agg: {"doc_count": imp},
            clicks_agg: {"doc_count": clk, position_agg: {"value": pos}},
        })
    return json.dumps(buckets)


# ---------------------------------------------------------------------------
# Overall CTR / clicks-per-search
# ---------------------------------------------------------------------------

class TestOverallMetrics:
    async def test_clicks_per_search_from_totals(self):
        r = await _result(compute_ubi_metrics(total_queries=100, total_clicks=25))
        assert r["clicks_per_search"] == 0.25

    async def test_clicks_per_search_can_exceed_one(self):
        # 300 clicks across 100 searches -> 3 clicks/search; NOT a CTR
        r = await _result(compute_ubi_metrics(total_queries=100, total_clicks=300))
        assert r["clicks_per_search"] == 3.0

    async def test_ctr_requires_queries_with_clicks(self):
        r = await _result(compute_ubi_metrics(total_queries=100, total_clicks=25))
        assert "ctr" not in r
        assert "ctr_note" in r

    async def test_bounded_ctr_computed(self):
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=300, queries_with_clicks=40
        ))
        # 40 of 100 searches had a click -> CTR 40%, bounded even with 300 clicks
        assert r["ctr"] == 0.40
        assert r["ctr_pct"] == "40.00%"

    async def test_zero_queries_returns_note(self):
        r = await _result(compute_ubi_metrics(total_queries=0, total_clicks=0))
        assert r["clicks_per_search"] == 0.0
        assert "note" in r

    async def test_totals_echoed_back(self):
        r = await _result(compute_ubi_metrics(total_queries=200, total_clicks=40))
        assert r["total_queries"] == 200
        assert r["total_clicks"] == 40


# ---------------------------------------------------------------------------
# Zero-click rate
# ---------------------------------------------------------------------------

class TestZeroClickRate:
    async def test_zero_click_rate_is_one_minus_ctr(self):
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=30, queries_with_clicks=20
        ))
        assert r["ctr"] == 0.20
        assert r["zero_click_rate"] == 0.80
        assert r["zero_click_rate_pct"] == "80.00%"
        assert r["queries_with_clicks"] == 20
        assert r["queries_without_clicks"] == 80

    async def test_all_queries_have_clicks(self):
        r = await _result(compute_ubi_metrics(
            total_queries=50, total_clicks=50, queries_with_clicks=50
        ))
        assert r["ctr"] == 1.0
        assert r["zero_click_rate"] == 0.0

    async def test_no_queries_with_clicks(self):
        r = await _result(compute_ubi_metrics(
            total_queries=50, total_clicks=0, queries_with_clicks=0
        ))
        assert r["ctr"] == 0.0
        assert r["zero_click_rate"] == 1.0

    async def test_omitted_queries_with_clicks_skips_ctr_and_zero_click_rate(self):
        r = await _result(compute_ubi_metrics(total_queries=100, total_clicks=10))
        assert "zero_click_rate" not in r
        assert "ctr" not in r


# ---------------------------------------------------------------------------
# Per-query breakdown from raw buckets
# ---------------------------------------------------------------------------

class TestPerQueryBreakdown:
    async def test_bounded_per_query_ctr(self):
        # laptop: 100 searches, 1 search with clicks -> CTR 1%
        sv = _sv_buckets([("laptop", 100), ("phone", 50)])
        clk = _click_buckets([("laptop", 20, 1), ("phone", 10, 5)])
        r = await _result(compute_ubi_metrics(
            total_queries=150, total_clicks=30,
            search_volume_buckets=sv, click_buckets=clk,
        ))
        by_query = {q["query"]: q for q in r["top_queries_by_ctr"]}
        assert by_query["laptop"]["search_volume"] == 100
        assert by_query["laptop"]["searches_with_clicks"] == 1
        assert by_query["laptop"]["total_clicks"] == 20
        assert by_query["laptop"]["ctr"] == 0.01
        assert by_query["laptop"]["ctr_pct"] == "1.00%"
        # clicks_per_search is the unbounded metric: 20 clicks / 100 searches
        assert by_query["laptop"]["clicks_per_search"] == 0.2
        assert by_query["laptop"]["zero_click_rate"] == 0.99

    async def test_searches_with_clicks_used_for_bounded_ctr(self):
        sv = _sv_buckets([("laptop", 200)])
        # 25 clicks spread over 2 searches (query_ids) -> bounded CTR uses the 2
        clk = _click_buckets([("laptop", 25, 2)])
        r = await _result(compute_ubi_metrics(
            total_queries=200, total_clicks=25,
            search_volume_buckets=sv, click_buckets=clk,
        ))
        laptop = next(q for q in r["top_queries_by_ctr"] if q["query"] == "laptop")
        assert laptop["total_clicks"] == 25
        assert laptop["searches_with_clicks"] == 2
        assert laptop["ctr"] == round(2 / 200, 4)            # bounded CTR
        assert laptop["clicks_per_search"] == 0.13           # 25/200=0.125 -> half-up 2dp

    async def test_results_sorted_by_ctr_descending(self):
        sv = _sv_buckets([("a", 100), ("b", 100), ("c", 100)])
        clk = _click_buckets([("a", 10, 10), ("b", 10, 5), ("c", 10, 30)])
        r = await _result(compute_ubi_metrics(
            total_queries=300, total_clicks=30,
            search_volume_buckets=sv, click_buckets=clk,
        ))
        ctrs = [q["ctr"] for q in r["top_queries_by_ctr"]]
        assert ctrs == sorted(ctrs, reverse=True)

    async def test_min_search_volume_filters_ranking(self):
        sv = _sv_buckets([("popular", 100), ("rare", 2)])
        clk = _click_buckets([("popular", 5, 5), ("rare", 1, 1)])
        r = await _result(compute_ubi_metrics(
            total_queries=102, total_clicks=6,
            search_volume_buckets=sv, click_buckets=clk,
            min_search_volume=5,
        ))
        ranked_queries = {q["query"] for q in r["top_queries_by_ctr"]}
        assert "rare" not in ranked_queries
        assert "popular" in ranked_queries
        assert r["per_query_count"] == 2      # count is unfiltered
        assert r["ranked_query_count"] == 1   # ranking is filtered

    async def test_query_with_searches_but_no_clicks_has_zero_ctr(self):
        sv = _sv_buckets([("laptop", 100), ("tablet", 50)])
        clk = _click_buckets([("laptop", 10, 3)])
        r = await _result(compute_ubi_metrics(
            total_queries=150, total_clicks=10,
            search_volume_buckets=sv, click_buckets=clk,
        ))
        tablet = next(q for q in r["top_queries_by_ctr"] if q["query"] == "tablet")
        assert tablet["ctr"] == 0.0
        assert tablet["searches_with_clicks"] == 0
        assert tablet["total_clicks"] == 0

    async def test_query_with_clicks_but_no_search_volume_bucket_has_zero_volume(self):
        sv = _sv_buckets([("laptop", 100)])
        clk = _click_buckets([("laptop", 5, 2), ("unknown_query", 3, 1)])
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=8,
            search_volume_buckets=sv, click_buckets=clk,
        ))
        unknown = next((q for q in r["top_queries_by_ctr"] if q["query"] == "unknown_query"), None)
        assert unknown is not None
        assert unknown["search_volume"] == 0
        assert unknown["ctr"] == 0.0

    async def test_clicks_without_query_text_echoed(self):
        sv = _sv_buckets([("laptop", 100)])
        clk = _click_buckets([("laptop", 10, 3)])
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=38,
            search_volume_buckets=sv, click_buckets=clk,
            clicks_without_query_text=28,
        ))
        assert r["clicks_without_query_text"] == 28

    async def test_per_query_count_returned(self):
        sv = _sv_buckets([("a", 10), ("b", 20)])
        clk = _click_buckets([("a", 2, 1)])
        r = await _result(compute_ubi_metrics(
            total_queries=30, total_clicks=2,
            search_volume_buckets=sv, click_buckets=clk,
        ))
        assert r["per_query_count"] == 2

    async def test_custom_searches_with_clicks_agg_name(self):
        sv = _sv_buckets([("laptop", 100)])
        clk = _click_buckets([("laptop", 10, 1)], swc_agg="uniq_searches")
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=10,
            search_volume_buckets=sv, click_buckets=clk,
            searches_with_clicks_agg="uniq_searches",
        ))
        laptop = next(q for q in r["top_queries_by_ctr"] if q["query"] == "laptop")
        assert laptop["searches_with_clicks"] == 1
        assert laptop["ctr"] == 0.01

    async def test_missing_swc_subagg_defaults_to_zero(self):
        sv = _sv_buckets([("laptop", 100)])
        clk = json.dumps([{"key": "laptop", "doc_count": 5}])  # no cardinality sub-agg
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=5,
            search_volume_buckets=sv, click_buckets=clk,
        ))
        laptop = next(q for q in r["top_queries_by_ctr"] if q["query"] == "laptop")
        assert laptop["searches_with_clicks"] == 0
        assert laptop["total_clicks"] == 5
        assert laptop["ctr"] == 0.0


# ---------------------------------------------------------------------------
# Malformed buckets
# ---------------------------------------------------------------------------

class TestMalformedBuckets:
    async def test_invalid_search_volume_buckets_json_returns_error(self):
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=5,
            search_volume_buckets="not-json",
            click_buckets="[]",
        ))
        assert "error" in r

    async def test_invalid_click_buckets_json_returns_error(self):
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=5,
            search_volume_buckets="[]",
            click_buckets="{not-a-list}",
        ))
        assert "error" in r

    async def test_non_list_search_volume_buckets_returns_error(self):
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=5,
            search_volume_buckets='{"key": "laptop"}',  # dict, not list
            click_buckets="[]",
        ))
        assert "error" in r


# ---------------------------------------------------------------------------
# Document CTR
# ---------------------------------------------------------------------------

class TestSingleDocumentCTR:
    async def test_basic_document_ctr(self):
        r = await _result(compute_document_ctr(
            document_id="B07ZCRSVBB", impressions=200, clicks=50, avg_click_position=2.5
        ))
        assert r["document_id"] == "B07ZCRSVBB"
        assert r["impressions"] == 200
        assert r["clicks"] == 50
        assert r["ctr"] == 0.25
        assert r["ctr_pct"] == "25.00%"
        assert r["avg_click_position"] == 2.5

    async def test_zero_impressions_gives_zero_ctr(self):
        r = await _result(compute_document_ctr(
            document_id="doc1", impressions=0, clicks=0
        ))
        assert r["ctr"] == 0.0

    async def test_missing_counts_returns_error(self):
        r = await _result(compute_document_ctr(document_id="doc1"))
        assert "error" in r

    async def test_avg_position_none_preserved(self):
        r = await _result(compute_document_ctr(
            document_id="doc1", impressions=10, clicks=0
        ))
        assert r["avg_click_position"] is None


class TestMultiDocumentCTR:
    async def test_ranked_by_ctr(self):
        buckets = _doc_buckets([
            ("low", 100, 5, 8.0),    # 5%
            ("high", 100, 40, 1.2),  # 40%
            ("mid", 100, 20, 3.0),   # 20%
        ])
        r = await _result(compute_document_ctr(document_buckets=buckets))
        order = [d["document_id"] for d in r["top_documents_by_ctr"]]
        assert order == ["high", "mid", "low"]
        assert r["document_count"] == 3

    async def test_min_impressions_filter(self):
        buckets = _doc_buckets([
            ("popular", 100, 10, 2.0),
            ("rare", 2, 2, 1.0),
        ])
        r = await _result(compute_document_ctr(document_buckets=buckets, min_impressions=5))
        ids = {d["document_id"] for d in r["top_documents_by_ctr"]}
        assert ids == {"popular"}
        assert r["document_count"] == 1

    async def test_invalid_document_buckets_json_returns_error(self):
        r = await _result(compute_document_ctr(document_buckets="not-json"))
        assert "error" in r
