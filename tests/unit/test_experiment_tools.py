"""
Unit tests for experiment tools.

Tests critical paths for experiment operations including:
- Results aggregation (pairwise and pointwise)
- Input parsing and validation
- Error handling
"""

import json
from unittest.mock import patch

import pytest

from src.tools.art.experiment_tools import (
    aggregate_experiment_results,
)

pytestmark = pytest.mark.unit


def _pairwise_experiment(experiment_id="exp1", status="COMPLETED", results=None, search_config_list=None):
    """Build a minimal pairwise experiment document."""
    doc = {
        "id": experiment_id,
        "type": "PAIRWISE_COMPARISON",
        "status": status,
        "searchConfigurationList": search_config_list or ["config1", "config2"],
        "results": results if results is not None else [],
    }
    if status == "ERROR":
        doc["errorMessage"] = "Test error"
    return json.dumps(doc)


def _pointwise_experiment(experiment_id="exp1", status="COMPLETED"):
    """Build a minimal pointwise experiment document."""
    doc = {"id": experiment_id, "type": "POINTWISE_EVALUATION", "status": status}
    if status == "ERROR":
        doc["errorMessage"] = "Test error"
    return json.dumps(doc)


def _eval_results(hits):
    """Wrap hits in an OpenSearch-style search response."""
    return json.dumps({"hits": {"hits": [{"_source": h} for h in hits]}})


class TestAggregateExperimentResults:
    """Tests for aggregate_experiment_results function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    # --- Input validation ---

    @pytest.mark.asyncio
    async def test_invalid_json_experiment_data(self):
        """Returns error for malformed experiment_data JSON."""
        result = await aggregate_experiment_results("not-json")
        result_data = json.loads(result)
        assert "error" in result_data
        assert "Invalid JSON" in result_data["error"]

    @pytest.mark.asyncio
    async def test_invalid_json_evaluation_results(self):
        """Returns error for malformed evaluation_results JSON."""
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results="not-json"
        )
        result_data = json.loads(result)
        assert "error" in result_data
        assert "Invalid JSON" in result_data["error"]

    # --- Status handling ---

    @pytest.mark.asyncio
    async def test_error_status(self):
        """Returns error details when experiment status is ERROR."""
        result = await aggregate_experiment_results(_pairwise_experiment(status="ERROR"))
        result_data = json.loads(result)
        assert result_data["status"] == "ERROR"
        assert "error_message" in result_data
        assert "No results available" in result_data["message"]

    @pytest.mark.asyncio
    async def test_pending_status(self):
        """Returns in-progress message for PENDING status."""
        result = await aggregate_experiment_results(_pairwise_experiment(status="PENDING"))
        result_data = json.loads(result)
        assert result_data["status"] == "PENDING"
        assert "still pending" in result_data["message"].lower()

    @pytest.mark.asyncio
    async def test_running_status(self):
        """Returns in-progress message for RUNNING status."""
        result = await aggregate_experiment_results(_pairwise_experiment(status="RUNNING"))
        result_data = json.loads(result)
        assert result_data["status"] == "RUNNING"
        assert "message" in result_data

    @pytest.mark.asyncio
    async def test_unknown_status(self):
        """Returns not-available message for unrecognised status."""
        result = await aggregate_experiment_results(_pairwise_experiment(status="CANCELLED"))
        result_data = json.loads(result)
        assert result_data["status"] == "CANCELLED"
        assert "Results not available" in result_data["message"]
        assert "CANCELLED" in result_data["message"]

    # --- Unsupported type ---

    @pytest.mark.asyncio
    async def test_unsupported_experiment_type(self):
        """Returns error for unknown experiment type."""
        doc = json.dumps({"id": "exp1", "type": "UNKNOWN_TYPE", "status": "COMPLETED"})
        result = await aggregate_experiment_results(doc)
        result_data = json.loads(result)
        assert "error" in result_data
        assert "Unsupported experiment type" in result_data["error"]

    # --- Pointwise: missing evaluation_results ---

    @pytest.mark.asyncio
    async def test_pointwise_missing_evaluation_results(self):
        """Returns a helpful error when evaluation_results is omitted for pointwise."""
        result = await aggregate_experiment_results(_pointwise_experiment())
        result_data = json.loads(result)
        assert "error" in result_data
        assert "evaluation_results is required" in result_data["error"]

    # --- Pairwise success ---

    @pytest.mark.asyncio
    async def test_pairwise_success(self):
        """Pairwise experiment with two queries aggregates correctly."""
        results = [
            {
                "query_text": "laptop",
                "metrics": [
                    {"metric": "jaccard", "value": 0.8},
                    {"metric": "rbo50", "value": 0.75},
                ],
                "snapshots": [
                    {"searchConfigurationId": "config1", "docIds": ["doc1", "doc2"]},
                    {"searchConfigurationId": "config2", "docIds": ["doc3", "doc4"]},
                ],
            },
            {
                "query_text": "phone",
                "metrics": [{"metric": "jaccard", "value": 0.6}],
                "snapshots": [
                    {"searchConfigurationId": "config1", "docIds": ["doc5"]},
                    {"searchConfigurationId": "config2", "docIds": ["doc6"]},
                ],
            },
        ]
        result = await aggregate_experiment_results(_pairwise_experiment(results=results))
        result_data = json.loads(result)

        assert result_data["experiment_id"] == "exp1"
        assert result_data["type"] == "PAIRWISE_COMPARISON"
        assert result_data["total_queries"] == 2
        assert "aggregate_metrics" in result_data
        assert "jaccard" in result_data["aggregate_metrics"]
        assert result_data["aggregate_metrics"]["jaccard"]["mean"] == 0.7
        assert "top_performing_queries" in result_data
        assert "per_query_results" in result_data

    @pytest.mark.asyncio
    async def test_pairwise_empty_results(self):
        """Pairwise experiment with no results returns appropriate message."""
        result = await aggregate_experiment_results(_pairwise_experiment(results=[]))
        result_data = json.loads(result)
        assert result_data["total_queries"] == 0
        assert "message" in result_data
        assert "No results found" in result_data["message"]

    @pytest.mark.asyncio
    async def test_pairwise_null_metric_values(self):
        """Null metric values are excluded from aggregation without error."""
        results = [
            {
                "query_text": "test",
                "metrics": [
                    {"metric": "jaccard", "value": None},
                    {"metric": "rbo50", "value": 0.5},
                ],
                "snapshots": [],
            }
        ]
        result = await aggregate_experiment_results(_pairwise_experiment(results=results))
        result_data = json.loads(result)
        assert result_data["total_queries"] == 1
        assert "rbo50" in result_data["aggregate_metrics"]
        assert "jaccard" not in result_data["aggregate_metrics"]

    @pytest.mark.asyncio
    async def test_pairwise_empty_metrics_list(self):
        """Empty metrics list produces no aggregate_metrics."""
        results = [{"query_text": "test", "metrics": [], "snapshots": []}]
        result = await aggregate_experiment_results(_pairwise_experiment(results=results))
        result_data = json.loads(result)
        assert result_data["aggregate_metrics"] == {}
        assert result_data["primary_metric"] is None

    @pytest.mark.asyncio
    async def test_pairwise_malformed_metric_objects(self):
        """Malformed metric objects (missing name or value) are handled without error."""
        results = [
            {
                "query_text": "test",
                "metrics": [
                    {"metric": "jaccard", "value": 0.8},
                    {"metric": "rbo50"},           # missing value → None, skipped for agg
                    {"value": 0.5},                # missing name → skipped entirely
                    {},                            # empty → skipped
                ],
                "snapshots": [],
            }
        ]
        result = await aggregate_experiment_results(_pairwise_experiment(results=results))
        result_data = json.loads(result)
        assert result_data["total_queries"] == 1
        assert "jaccard" in result_data["aggregate_metrics"]
        assert result_data["aggregate_metrics"]["jaccard"]["mean"] == 0.8

    @pytest.mark.asyncio
    async def test_pairwise_missing_query_text(self):
        """Missing query_text defaults to empty string."""
        results = [{"metrics": [{"metric": "jaccard", "value": 0.8}], "snapshots": []}]
        result = await aggregate_experiment_results(_pairwise_experiment(results=results))
        result_data = json.loads(result)
        assert result_data["per_query_results"][0]["query_text"] == ""

    @pytest.mark.asyncio
    async def test_pairwise_missing_snapshots(self):
        """Missing snapshots field defaults to empty list."""
        results = [{"query_text": "test", "metrics": [{"metric": "jaccard", "value": 0.8}]}]
        result = await aggregate_experiment_results(_pairwise_experiment(results=results))
        result_data = json.loads(result)
        assert result_data["per_query_results"][0]["snapshots"] == []

    @pytest.mark.asyncio
    async def test_pairwise_single_value_stdev_zero(self):
        """Single-query experiment reports std_dev of 0."""
        results = [
            {"query_text": "test", "metrics": [{"metric": "jaccard", "value": 0.8}], "snapshots": []}
        ]
        result = await aggregate_experiment_results(_pairwise_experiment(results=results))
        result_data = json.loads(result)
        assert result_data["aggregate_metrics"]["jaccard"]["std_dev"] == 0

    @pytest.mark.asyncio
    async def test_pairwise_negative_metric_values(self):
        """Negative metric values are handled in min/max correctly."""
        results = [
            {"query_text": "t1", "metrics": [{"metric": "jaccard", "value": -0.5}], "snapshots": []},
            {"query_text": "t2", "metrics": [{"metric": "jaccard", "value": 0.8}], "snapshots": []},
        ]
        result = await aggregate_experiment_results(_pairwise_experiment(results=results))
        result_data = json.loads(result)
        assert result_data["aggregate_metrics"]["jaccard"]["min"] == -0.5
        assert result_data["aggregate_metrics"]["jaccard"]["max"] == 0.8

    @pytest.mark.asyncio
    async def test_pairwise_all_same_values(self):
        """All-equal metrics produce zero std_dev and equal mean/median/min/max."""
        results = [
            {"query_text": f"q{i}", "metrics": [{"metric": "jaccard", "value": 0.5}], "snapshots": []}
            for i in range(3)
        ]
        result = await aggregate_experiment_results(_pairwise_experiment(results=results))
        result_data = json.loads(result)
        agg = result_data["aggregate_metrics"]["jaccard"]
        assert agg["mean"] == agg["median"] == agg["min"] == agg["max"] == 0.5
        assert agg["std_dev"] == 0

    @pytest.mark.asyncio
    async def test_pairwise_large_experiment(self):
        """500-query pairwise experiment aggregates without error."""
        results = [
            {
                "query_text": f"query_{i}",
                "metrics": [
                    {"metric": "jaccard", "value": 0.5 + (i % 10) * 0.05},
                    {"metric": "rbo50", "value": 0.4 + (i % 10) * 0.05},
                ],
                "snapshots": [
                    {"searchConfigurationId": "config1", "docIds": [f"doc{j}" for j in range(10)]},
                    {"searchConfigurationId": "config2", "docIds": [f"doc{j+10}" for j in range(10)]},
                ],
            }
            for i in range(500)
        ]
        result = await aggregate_experiment_results(_pairwise_experiment(results=results))
        result_data = json.loads(result)
        assert result_data["total_queries"] == 500
        assert len(result_data["per_query_results"]) == 500

    # --- Pointwise success ---

    @pytest.mark.asyncio
    async def test_pointwise_success(self):
        """Pointwise experiment with two queries aggregates correctly."""
        hits = [
            {
                "searchText": "laptop",
                "metrics": [{"metric": "NDCG@10", "value": 0.85}],
                "documentIds": ["doc1", "doc2"],
                "searchConfigurationId": "config1",
            },
            {
                "searchText": "phone",
                "metrics": [{"metric": "NDCG@10", "value": 0.70}],
                "documentIds": ["doc3"],
                "searchConfigurationId": "config1",
            },
        ]
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results=_eval_results(hits)
        )
        result_data = json.loads(result)

        assert result_data["experiment_id"] == "exp1"
        assert result_data["type"] == "POINTWISE_EVALUATION"
        assert result_data["total_queries"] == 2
        assert "NDCG@10" in result_data["aggregate_metrics"]
        assert result_data["aggregate_metrics"]["NDCG@10"]["mean"] == round((0.85 + 0.70) / 2, 4)

    @pytest.mark.asyncio
    async def test_pointwise_empty_hits(self):
        """Empty evaluation results returns appropriate message."""
        result = await aggregate_experiment_results(
            _pointwise_experiment(),
            evaluation_results=json.dumps({"hits": {"hits": []}}),
        )
        result_data = json.loads(result)
        assert result_data["total_queries"] == 0
        assert "message" in result_data

    @pytest.mark.asyncio
    async def test_pointwise_direct_array_input(self):
        """evaluation_results can be a plain JSON array of source documents."""
        hits = [
            {
                "searchText": "laptop",
                "metrics": [{"metric": "NDCG@10", "value": 0.8}],
                "documentIds": ["doc1"],
                "searchConfigurationId": "config1",
            }
        ]
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results=json.dumps(hits)
        )
        result_data = json.loads(result)
        assert result_data["total_queries"] == 1

    @pytest.mark.asyncio
    async def test_pointwise_null_metric_values(self):
        """Null metric values are excluded from pointwise aggregation."""
        hits = [
            {
                "searchText": "test",
                "metrics": [
                    {"metric": "NDCG@10", "value": None},
                    {"metric": "MRR", "value": 0.5},
                ],
                "documentIds": ["doc1"],
                "searchConfigurationId": "config1",
            }
        ]
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results=_eval_results(hits)
        )
        result_data = json.loads(result)
        assert "MRR" in result_data["aggregate_metrics"]
        assert "NDCG@10" not in result_data["aggregate_metrics"]

    @pytest.mark.asyncio
    async def test_pointwise_empty_metrics_list(self):
        """Empty metrics list produces no aggregate_metrics."""
        hits = [{"searchText": "test", "metrics": [], "documentIds": ["doc1"], "searchConfigurationId": "config1"}]
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results=_eval_results(hits)
        )
        result_data = json.loads(result)
        assert result_data["aggregate_metrics"] == {}

    @pytest.mark.asyncio
    async def test_pointwise_missing_search_text(self):
        """Missing searchText defaults to empty string."""
        hits = [{"metrics": [{"metric": "NDCG@10", "value": 0.8}], "documentIds": ["doc1"], "searchConfigurationId": "config1"}]
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results=_eval_results(hits)
        )
        result_data = json.loads(result)
        assert result_data["per_query_results"][0]["query_text"] == ""

    @pytest.mark.asyncio
    async def test_pointwise_missing_document_ids(self):
        """Missing documentIds defaults to empty list."""
        hits = [{"searchText": "test", "metrics": [{"metric": "NDCG@10", "value": 0.8}], "searchConfigurationId": "config1"}]
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results=_eval_results(hits)
        )
        result_data = json.loads(result)
        assert result_data["per_query_results"][0]["document_ids"] == []

    @pytest.mark.asyncio
    async def test_pointwise_missing_search_configuration_id(self):
        """Missing searchConfigurationId defaults to empty string."""
        hits = [{"searchText": "test", "metrics": [{"metric": "NDCG@10", "value": 0.8}], "documentIds": ["doc1"]}]
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results=_eval_results(hits)
        )
        result_data = json.loads(result)
        assert result_data["per_query_results"][0]["search_configuration_id"] == ""

    @pytest.mark.asyncio
    async def test_pointwise_fallback_primary_metric(self):
        """Falls back to first available metric when NDCG@10 is absent."""
        hits = [{"searchText": "test", "metrics": [{"metric": "MRR", "value": 0.8}], "documentIds": ["doc1"], "searchConfigurationId": "config1"}]
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results=_eval_results(hits)
        )
        result_data = json.loads(result)
        assert result_data["primary_metric"] == "MRR"

    @pytest.mark.asyncio
    async def test_pointwise_single_value_stdev_zero(self):
        """Single-query pointwise experiment reports std_dev of 0."""
        hits = [{"searchText": "test", "metrics": [{"metric": "NDCG@10", "value": 0.8}], "documentIds": ["doc1"], "searchConfigurationId": "config1"}]
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results=_eval_results(hits)
        )
        result_data = json.loads(result)
        assert result_data["aggregate_metrics"]["NDCG@10"]["std_dev"] == 0

    @pytest.mark.asyncio
    async def test_pointwise_malformed_metric_objects(self):
        """Malformed metric objects are handled without error."""
        hits = [
            {
                "searchText": "test",
                "metrics": [
                    {"metric": "NDCG@10", "value": 0.8},
                    {"metric": "MRR"},       # missing value
                    {"value": 0.5},          # missing name
                    {},                      # empty
                ],
                "documentIds": ["doc1"],
                "searchConfigurationId": "config1",
            }
        ]
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results=_eval_results(hits)
        )
        result_data = json.loads(result)
        assert "NDCG@10" in result_data["aggregate_metrics"]
        assert result_data["aggregate_metrics"]["NDCG@10"]["mean"] == 0.8

    @pytest.mark.asyncio
    async def test_pointwise_large_experiment(self):
        """1000-query pointwise experiment aggregates without error."""
        hits = [
            {
                "searchText": f"query_{i}",
                "metrics": [{"metric": "NDCG@10", "value": 0.5 + (i % 10) * 0.05}],
                "documentIds": [f"doc{j}" for j in range(10)],
                "searchConfigurationId": "config1",
            }
            for i in range(1000)
        ]
        result = await aggregate_experiment_results(
            _pointwise_experiment(), evaluation_results=_eval_results(hits)
        )
        result_data = json.loads(result)
        assert result_data["total_queries"] == 1000
        assert len(result_data["per_query_results"]) == 1000

    # --- Concurrent calls ---

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Multiple concurrent calls complete successfully without interference."""
        import asyncio

        results_data = [
            {"query_text": "test", "metrics": [{"metric": "jaccard", "value": 0.8}], "snapshots": []}
        ]
        exp = _pairwise_experiment(results=results_data)
        results = await asyncio.gather(
            aggregate_experiment_results(exp),
            aggregate_experiment_results(exp),
            aggregate_experiment_results(exp),
        )
        for result in results:
            assert json.loads(result)["total_queries"] == 1
