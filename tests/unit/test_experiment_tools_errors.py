"""
Unit tests for experiment tools error scenarios.

Tests error paths and edge cases for experiment operations including:
- Invalid input handling
- Experiment not found / wrong status scenarios
- Empty results handling
"""

import json
from unittest.mock import patch

import pytest

from src.tools.art.experiment_tools import (
    aggregate_experiment_results,
)

pytestmark = pytest.mark.unit


def _pairwise_doc(status="COMPLETED", results=None, error_message=None):
    doc = {
        "id": "exp1",
        "type": "PAIRWISE_COMPARISON",
        "status": status,
        "searchConfigurationList": ["config1", "config2"],
        "results": results if results is not None else [],
    }
    if error_message:
        doc["errorMessage"] = error_message
    return json.dumps(doc)


class TestExperimentToolsErrors:
    """Test experiment tool error scenarios."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Completed experiment with no results returns a message, not an error."""
        result = await aggregate_experiment_results(_pairwise_doc(results=[]))
        result_data = json.loads(result)

        assert result_data["experiment_id"] == "exp1"
        assert result_data["type"] == "PAIRWISE_COMPARISON"
        assert result_data["total_queries"] == 0
        assert "message" in result_data
        assert "No results found" in result_data["message"]

    @pytest.mark.asyncio
    async def test_invalid_experiment_data(self):
        """Invalid JSON in experiment_data returns a descriptive error."""
        result = await aggregate_experiment_results("{invalid json}")
        result_data = json.loads(result)

        assert "error" in result_data
        assert "Invalid JSON" in result_data["error"]

    @pytest.mark.asyncio
    async def test_error_status(self):
        """Experiment with ERROR status returns error details."""
        result = await aggregate_experiment_results(
            _pairwise_doc(status="ERROR", error_message="Experiment execution failed")
        )
        result_data = json.loads(result)

        assert result_data["status"] == "ERROR"
        assert "error_message" in result_data
        assert result_data["error_message"] == "Experiment execution failed"
        assert "message" in result_data
        assert "No results available" in result_data["message"]

    @pytest.mark.asyncio
    async def test_pending_status(self):
        """Experiment with PENDING status returns an in-progress message."""
        result = await aggregate_experiment_results(_pairwise_doc(status="PENDING"))
        result_data = json.loads(result)

        assert result_data["status"] == "PENDING"
        assert "message" in result_data
        assert "still pending" in result_data["message"].lower()

    @pytest.mark.asyncio
    async def test_unknown_status(self):
        """Experiment with an unrecognised status returns a not-available message."""
        result = await aggregate_experiment_results(_pairwise_doc(status="CANCELLED"))
        result_data = json.loads(result)

        assert result_data["status"] == "CANCELLED"
        assert "message" in result_data
        assert "Results not available" in result_data["message"]
        assert "CANCELLED" in result_data["message"]
