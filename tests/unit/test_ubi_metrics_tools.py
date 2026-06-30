"""
Unit tests for compute_ubi_metrics.

Covers:
- Global CTR from total_queries / total_clicks
- Zero-click rate when queries_with_clicks is provided
- Zero total_queries edge case
- Per-query CTR computed from raw aggregation buckets (the tool joins and divides)
- Queries appearing only in impressions or only in clicks
- Skipped buckets when top_hits sub-agg is missing or malformed
- Invalid JSON inputs return an error key rather than raising
"""

import json

import pytest

from tools.art.ubi_metrics_tools import compute_ubi_metrics

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _result(coro) -> dict:
    return json.loads(await coro)


def _imp_buckets(pairs: list[tuple[str, int]]) -> str:
    """Build an impression bucket list: [(query_text, doc_count), ...]."""
    return json.dumps([{"key": q, "doc_count": c} for q, c in pairs])


def _click_buckets(
    pairs: list[tuple[str, str, int]],
    agg_name: str = "query_text",
    field: str = "user_query",
) -> str:
    """Build a click-by-query_id bucket list: [(query_id, query_text, doc_count), ...]."""
    buckets = []
    for qid, qt, count in pairs:
        buckets.append({
            "key": qid,
            "doc_count": count,
            agg_name: {"hits": {"hits": [{"_source": {field: qt}}]}},
        })
    return json.dumps(buckets)


# ---------------------------------------------------------------------------
# Global CTR
# ---------------------------------------------------------------------------

class TestGlobalCTR:
    async def test_basic_ctr(self):
        r = await _result(compute_ubi_metrics(total_queries=100, total_clicks=25))
        assert r["overall_ctr"] == 0.25
        assert r["overall_ctr_pct"] == "25.00%"

    async def test_zero_queries_returns_zero_ctr(self):
        r = await _result(compute_ubi_metrics(total_queries=0, total_clicks=0))
        assert r["overall_ctr"] == 0.0
        assert "note" in r

    async def test_zero_clicks_gives_zero_ctr(self):
        r = await _result(compute_ubi_metrics(total_queries=50, total_clicks=0))
        assert r["overall_ctr"] == 0.0

    async def test_ctr_rounded_to_four_decimal_places(self):
        r = await _result(compute_ubi_metrics(total_queries=3, total_clicks=1))
        assert r["overall_ctr"] == round(1 / 3, 4)

    async def test_totals_echoed_back(self):
        r = await _result(compute_ubi_metrics(total_queries=200, total_clicks=40))
        assert r["total_queries"] == 200
        assert r["total_clicks"] == 40


# ---------------------------------------------------------------------------
# Zero-click rate
# ---------------------------------------------------------------------------

class TestZeroClickRate:
    async def test_zero_click_rate_computed(self):
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=30, queries_with_clicks=20
        ))
        assert r["zero_click_rate"] == 0.80
        assert r["zero_click_rate_pct"] == "80.00%"
        assert r["queries_with_clicks"] == 20
        assert r["queries_without_clicks"] == 80

    async def test_all_queries_have_clicks(self):
        r = await _result(compute_ubi_metrics(
            total_queries=50, total_clicks=50, queries_with_clicks=50
        ))
        assert r["zero_click_rate"] == 0.0

    async def test_no_queries_with_clicks(self):
        r = await _result(compute_ubi_metrics(
            total_queries=50, total_clicks=0, queries_with_clicks=0
        ))
        assert r["zero_click_rate"] == 1.0

    async def test_omitted_queries_with_clicks_skips_zero_click_rate(self):
        r = await _result(compute_ubi_metrics(total_queries=100, total_clicks=10))
        assert "zero_click_rate" not in r


# ---------------------------------------------------------------------------
# Per-query CTR from raw buckets
# ---------------------------------------------------------------------------

class TestPerQueryCTR:
    async def test_basic_per_query_ctr(self):
        imp = _imp_buckets([("laptop", 100), ("phone", 50)])
        clk = _click_buckets([("qid-1", "laptop", 20), ("qid-2", "phone", 10)])
        r = await _result(compute_ubi_metrics(
            total_queries=150, total_clicks=30,
            impression_buckets=imp, click_query_id_buckets=clk,
        ))
        by_query = {q["query"]: q for q in r["top_queries_by_ctr"]}
        assert by_query["laptop"]["ctr"] == 0.2
        assert by_query["laptop"]["ctr_pct"] == "20.00%"
        assert by_query["phone"]["ctr"] == 0.2

    async def test_multiple_query_ids_for_same_query_text_are_summed(self):
        imp = _imp_buckets([("laptop", 200)])
        # Two different query_ids both for "laptop"
        clk = _click_buckets([("qid-1", "laptop", 10), ("qid-2", "laptop", 15)])
        r = await _result(compute_ubi_metrics(
            total_queries=200, total_clicks=25,
            impression_buckets=imp, click_query_id_buckets=clk,
        ))
        laptop = next(q for q in r["top_queries_by_ctr"] if q["query"] == "laptop")
        assert laptop["clicks"] == 25
        assert laptop["ctr"] == round(25 / 200, 4)

    async def test_results_sorted_by_ctr_descending(self):
        imp = _imp_buckets([("a", 100), ("b", 100), ("c", 100)])
        clk = _click_buckets([("q1", "a", 50), ("q2", "b", 10), ("q3", "c", 30)])
        r = await _result(compute_ubi_metrics(
            total_queries=300, total_clicks=90,
            impression_buckets=imp, click_query_id_buckets=clk,
        ))
        ctrs = [q["ctr"] for q in r["top_queries_by_ctr"]]
        assert ctrs == sorted(ctrs, reverse=True)

    async def test_query_with_impressions_but_no_clicks_has_zero_ctr(self):
        imp = _imp_buckets([("laptop", 100), ("tablet", 50)])
        clk = _click_buckets([("qid-1", "laptop", 10)])
        r = await _result(compute_ubi_metrics(
            total_queries=150, total_clicks=10,
            impression_buckets=imp, click_query_id_buckets=clk,
        ))
        tablet = next(q for q in r["top_queries_by_ctr"] if q["query"] == "tablet")
        assert tablet["ctr"] == 0.0
        assert tablet["clicks"] == 0

    async def test_query_with_clicks_but_no_impression_bucket_has_zero_impressions(self):
        imp = _imp_buckets([("laptop", 100)])
        clk = _click_buckets([("qid-1", "laptop", 5), ("qid-2", "unknown_query", 3)])
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=8,
            impression_buckets=imp, click_query_id_buckets=clk,
        ))
        unknown = next((q for q in r["top_queries_by_ctr"] if q["query"] == "unknown_query"), None)
        assert unknown is not None
        assert unknown["impressions"] == 0
        assert unknown["ctr"] == 0.0

    async def test_per_query_count_returned(self):
        imp = _imp_buckets([("a", 10), ("b", 20)])
        clk = _click_buckets([("q1", "a", 2)])
        r = await _result(compute_ubi_metrics(
            total_queries=30, total_clicks=2,
            impression_buckets=imp, click_query_id_buckets=clk,
        ))
        assert r["per_query_count"] == 2

    async def test_custom_field_and_agg_name(self):
        imp = _imp_buckets([("laptop", 100)])
        clk = json.dumps([{
            "key": "qid-1",
            "doc_count": 10,
            "my_agg": {"hits": {"hits": [{"_source": {"search_text": "laptop"}}]}},
        }])
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=10,
            impression_buckets=imp, click_query_id_buckets=clk,
            query_text_field="search_text",
            click_query_text_agg="my_agg",
        ))
        laptop = next(q for q in r["top_queries_by_ctr"] if q["query"] == "laptop")
        assert laptop["ctr"] == 0.1


# ---------------------------------------------------------------------------
# Skipped / malformed buckets
# ---------------------------------------------------------------------------

class TestMalformedBuckets:
    async def test_bucket_missing_top_hits_agg_is_skipped(self):
        imp = _imp_buckets([("laptop", 100)])
        clk = json.dumps([{"key": "qid-1", "doc_count": 5}])  # no top_hits sub-agg
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=5,
            impression_buckets=imp, click_query_id_buckets=clk,
        ))
        assert r.get("skipped_click_buckets") == 1
        assert "skipped_reason" in r

    async def test_invalid_impression_buckets_json_returns_error(self):
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=5,
            impression_buckets="not-json",
            click_query_id_buckets="[]",
        ))
        assert "error" in r

    async def test_invalid_click_buckets_json_returns_error(self):
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=5,
            impression_buckets="[]",
            click_query_id_buckets="{not-a-list}",
        ))
        assert "error" in r

    async def test_non_list_impression_buckets_returns_error(self):
        r = await _result(compute_ubi_metrics(
            total_queries=100, total_clicks=5,
            impression_buckets='{"key": "laptop"}',  # dict, not list
            click_query_id_buckets="[]",
        ))
        assert "error" in r
