# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
Experiment Tools
Tools for aggregating search relevance experiment results.
Experiment data is retrieved via MCP tools (GetExperimentTool, SearchIndexTool)
and passed to this tool for aggregation and analysis.
"""

import json
import statistics

from utils.logging_helpers import (
    get_logger,
    log_error_event,
    log_info_event,
    log_warning_event,
)
from utils.monitored_tool import monitored_tool
from utils.tool_utils import format_tool_error

logger = get_logger(__name__)


@monitored_tool(
    name="AggregateExperimentResultsTool",
    description=(
        "Aggregates and analyzes results from an offline evaluation experiment. "
        "Pass experiment_data as the JSON output from GetExperimentTool. "
        "For POINTWISE_EVALUATION experiments, also pass evaluation_results as the "
        "JSON output from SearchIndexTool querying the search-relevance-evaluation-result "
        "index filtered by experimentId. "
        "Computes aggregate statistics (mean, median, std dev), identifies top/bottom "
        "performing queries, and returns per-query details."
    ),
)
def aggregate_experiment_results(
    experiment_data: str,
    evaluation_results: str = "",
) -> str:
    """
    Aggregates experiment results with computed metrics and per-query analysis.
    Handles both PAIRWISE_COMPARISON and POINTWISE_EVALUATION experiments.

    Args:
        experiment_data: JSON string of the experiment document from GetExperimentTool.
            Must contain 'type', 'status', and for pairwise experiments, 'results'.
        evaluation_results: JSON string of evaluation result hits from SearchIndexTool
            on the search-relevance-evaluation-result index. Required for
            POINTWISE_EVALUATION experiments.

    Returns:
        str: JSON string with aggregate metrics, top/bottom performers, and per-query results
    """
    try:
        try:
            exp_data = json.loads(experiment_data)
        except json.JSONDecodeError as e:
            return json.dumps(
                {"error": f"Invalid JSON in experiment_data: {str(e)}"}, indent=2
            )

        exp_type = exp_data.get("type", "")
        exp_status = exp_data.get("status", "")
        experiment_id = exp_data.get(
            "id", exp_data.get("experimentId", exp_data.get("experiment_id", "unknown"))
        )

        log_info_event(
            logger,
            f"[AggregateExperimentResultsTool] type={exp_type}, status={exp_status}",
            "tools.experiment.aggregate_start",
            experiment_id=experiment_id,
            exp_type=exp_type,
            exp_status=exp_status,
        )

        if exp_status == "ERROR":
            error_message = exp_data.get("errorMessage", "No error message provided")
            log_error_event(
                logger,
                f"[AggregateExperimentResultsTool] ✗ Experiment failed: {error_message}",
                "tools.experiment.aggregate_error_status",
                error=error_message,
                exc_info=False,
            )
            return json.dumps(
                {
                    "experiment_id": experiment_id,
                    "type": exp_type,
                    "status": "ERROR",
                    "error_message": error_message,
                    "message": "Experiment execution failed. No results available.",
                },
                indent=2,
            )

        if exp_status in ["PENDING", "RUNNING", "PROCESSING"]:
            log_info_event(
                logger,
                "[AggregateExperimentResultsTool] Experiment still processing.",
                "tools.experiment.aggregate_still_processing",
            )
            return json.dumps(
                {
                    "experiment_id": experiment_id,
                    "type": exp_type,
                    "status": exp_status,
                    "message": f"Experiment is still {exp_status.lower()}. Results not yet available. Please try again later.",
                },
                indent=2,
            )

        if exp_status != "COMPLETED":
            log_warning_event(
                logger,
                f"[AggregateExperimentResultsTool] Unexpected status: {exp_status}",
                "tools.experiment.aggregate_unexpected_status",
                exp_status=exp_status,
            )
            return json.dumps(
                {
                    "experiment_id": experiment_id,
                    "type": exp_type,
                    "status": exp_status,
                    "message": f"Results not available for status '{exp_status}'. Only COMPLETED experiments have results.",
                },
                indent=2,
            )

        if exp_type == "PAIRWISE_COMPARISON":
            log_info_event(
                logger,
                "[AggregateExperimentResultsTool] Processing as PAIRWISE_COMPARISON.",
                "tools.experiment.aggregate_pairwise",
            )
            return _aggregate_pairwise_results(experiment_id, exp_data)

        if exp_type == "POINTWISE_EVALUATION":
            if not evaluation_results:
                return json.dumps(
                    {
                        "error": (
                            "evaluation_results is required for POINTWISE_EVALUATION experiments. "
                            "Please search the search-relevance-evaluation-result index using "
                            "SearchIndexTool filtered by experimentId and pass the results here."
                        ),
                        "experiment_id": experiment_id,
                        "type": exp_type,
                    },
                    indent=2,
                )
            log_info_event(
                logger,
                "[AggregateExperimentResultsTool] Processing as POINTWISE_EVALUATION.",
                "tools.experiment.aggregate_pointwise",
            )
            return _aggregate_pointwise_results(experiment_id, evaluation_results)

        log_warning_event(
            logger,
            f"[AggregateExperimentResultsTool] Unsupported experiment type: {exp_type}",
            "tools.experiment.aggregate_unsupported_type",
            exp_type=exp_type,
        )
        return json.dumps(
            {
                "experiment_id": experiment_id,
                "type": exp_type,
                "error": f"Unsupported experiment type: {exp_type}",
                "message": "Only PAIRWISE_COMPARISON and POINTWISE_EVALUATION are currently supported.",
            },
            indent=2,
        )

    except Exception as e:
        error_msg = f"Error aggregating experiment results: {str(e)}"
        log_error_event(
            logger,
            f"[AggregateExperimentResultsTool] ERROR: {error_msg}",
            "tools.experiment.aggregate_exception",
            error=e,
        )
        return format_tool_error(error_msg)


def _aggregate_pairwise_results(experiment_id: str, exp_data: dict) -> str:
    """Aggregate results for PAIRWISE_COMPARISON experiments."""
    try:
        results = exp_data.get("results", [])
        search_config_ids = exp_data.get("searchConfigurationList", [])

        log_info_event(
            logger,
            f"[AggregateExperimentResultsTool] Found {len(results)} pairwise results",
            "tools.experiment.aggregate_pairwise_count",
            count=len(results),
        )

        if len(results) == 0:
            return json.dumps(
                {
                    "experiment_id": experiment_id,
                    "type": "PAIRWISE_COMPARISON",
                    "total_queries": 0,
                    "message": "No results found in pairwise experiment",
                },
                indent=2,
            )

        all_metrics: dict[str, list[float]] = {}
        per_query_results = []

        for result in results:
            query_text = result.get("query_text", "")
            metrics_list = result.get("metrics", [])
            snapshots = result.get("snapshots", [])

            query_metrics = {}
            for metric_obj in metrics_list:
                metric_name = metric_obj.get("metric")
                metric_value = metric_obj.get("value")
                query_metrics[metric_name] = metric_value

                if metric_name is not None and isinstance(metric_value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

            comparison_snapshots = []
            for snapshot in snapshots:
                comparison_snapshots.append(
                    {
                        "search_configuration_id": snapshot.get(
                            "searchConfigurationId", ""
                        ),
                        "document_ids": snapshot.get("docIds", []),
                        "num_documents": len(snapshot.get("docIds", [])),
                    }
                )

            per_query_results.append(
                {
                    "query_text": query_text,
                    "metrics": query_metrics,
                    "snapshots": comparison_snapshots,
                }
            )

        aggregate_metrics = _compute_aggregate_metrics(all_metrics)

        primary_metric = list(all_metrics.keys())[0] if all_metrics else None
        sorted_queries = sorted(
            per_query_results,
            key=lambda x: x["metrics"].get(primary_metric, 0) if primary_metric else 0,
            reverse=True,
        )

        for idx, query_result in enumerate(sorted_queries):
            query_result["performance_rank"] = idx + 1

        top_n = min(5, len(sorted_queries))
        bottom_n = min(5, len(sorted_queries))
        top_performers = [
            {"query": q["query_text"], "rank": q["performance_rank"], **q["metrics"]}
            for q in sorted_queries[:top_n]
        ]
        underperforming = [
            {"query": q["query_text"], "rank": q["performance_rank"], **q["metrics"]}
            for q in sorted_queries[-bottom_n:]
        ]

        log_info_event(
            logger,
            "[AggregateExperimentResultsTool] ✓ Successfully aggregated pairwise results",
            "tools.experiment.aggregate_pairwise_done",
            query_count=len(per_query_results),
        )
        return json.dumps(
            {
                "experiment_id": experiment_id,
                "type": "PAIRWISE_COMPARISON",
                "total_queries": len(per_query_results),
                "search_configurations_compared": search_config_ids,
                "primary_metric": primary_metric,
                "aggregate_metrics": aggregate_metrics,
                "top_performing_queries": top_performers,
                "underperforming_queries": underperforming,
                "per_query_results": sorted_queries,
            },
            indent=2,
        )

    except Exception as e:
        error_msg = f"Error processing pairwise results: {str(e)}"
        log_error_event(
            logger,
            "[AggregateExperimentResultsTool] ✗ ERROR in pairwise aggregation.",
            "tools.experiment.aggregate_pairwise_exception",
            error=e,
        )
        return format_tool_error(error_msg)


def _aggregate_pointwise_results(experiment_id: str, evaluation_results: str) -> str:
    """Aggregate results for POINTWISE_EVALUATION experiments."""
    try:
        try:
            results_data = json.loads(evaluation_results)
        except json.JSONDecodeError as e:
            return json.dumps(
                {"error": f"Invalid JSON in evaluation_results: {str(e)}"}, indent=2
            )

        # Accept either a full OpenSearch response ({hits: {hits: [...]}})
        # or a direct array of source documents
        if isinstance(results_data, list):
            hits = results_data
        else:
            hits = results_data.get("hits", {}).get("hits", [])

        log_info_event(
            logger,
            f"[AggregateExperimentResultsTool] Processing {len(hits)} pointwise results",
            "tools.experiment.aggregate_pointwise_count",
            hit_count=len(hits),
        )

        if not hits:
            log_warning_event(
                logger,
                "[AggregateExperimentResultsTool] No pointwise results found.",
                "tools.experiment.aggregate_pointwise_empty",
                experiment_id=experiment_id,
            )
            return json.dumps(
                {
                    "experiment_id": experiment_id,
                    "type": "POINTWISE_EVALUATION",
                    "total_queries": 0,
                    "message": "No results found for this pointwise evaluation experiment",
                },
                indent=2,
            )

        metrics_by_query: list[dict] = []
        all_metrics: dict[str, list[float]] = {}

        for hit in hits:
            # Accept either {_source: {...}} (OpenSearch hit) or flat source doc
            source = hit.get("_source", hit)
            query_text = source.get("searchText", "")
            metrics_list = source.get("metrics", [])

            query_metrics = {}
            for metric_obj in metrics_list:
                metric_name = metric_obj.get("metric")
                metric_value = metric_obj.get("value")
                query_metrics[metric_name] = metric_value

                if metric_name is not None and isinstance(metric_value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

            metrics_by_query.append(
                {
                    "query_text": query_text,
                    "metrics": query_metrics,
                    "document_ids": source.get("documentIds", []),
                    "search_configuration_id": source.get("searchConfigurationId", ""),
                    "timestamp": source.get("timestamp", ""),
                }
            )

        aggregate_metrics = _compute_aggregate_metrics(all_metrics)

        primary_metric = "NDCG@10"
        if primary_metric not in all_metrics and all_metrics:
            primary_metric = list(all_metrics.keys())[0]

        sorted_queries = sorted(
            metrics_by_query,
            key=lambda x: x["metrics"].get(primary_metric, 0),
            reverse=True,
        )
        for idx, query_result in enumerate(sorted_queries):
            query_result["performance_rank"] = idx + 1

        top_performers = [
            {"query": q["query_text"], "rank": q["performance_rank"], **q["metrics"]}
            for q in sorted_queries[:5]
        ]
        underperforming = [
            {"query": q["query_text"], "rank": q["performance_rank"], **q["metrics"]}
            for q in sorted_queries[-5:]
        ]

        log_info_event(
            logger,
            "[AggregateExperimentResultsTool] ✓ Successfully aggregated pointwise results",
            "tools.experiment.aggregate_pointwise_done",
            query_count=len(metrics_by_query),
        )
        return json.dumps(
            {
                "experiment_id": experiment_id,
                "type": "POINTWISE_EVALUATION",
                "total_queries": len(metrics_by_query),
                "primary_metric": primary_metric,
                "aggregate_metrics": aggregate_metrics,
                "top_performing_queries": top_performers,
                "underperforming_queries": underperforming,
                "per_query_results": sorted_queries,
            },
            indent=2,
        )

    except Exception as e:
        error_msg = f"Error processing pointwise results: {str(e)}"
        log_error_event(
            logger,
            "[AggregateExperimentResultsTool] ✗ ERROR in pointwise aggregation.",
            "tools.experiment.aggregate_pointwise_exception",
            error=e,
        )
        return format_tool_error(error_msg)


def _compute_aggregate_metrics(
    all_metrics: dict[str, list[float]],
) -> dict[str, dict]:
    """Compute mean, median, min, max, std_dev for each metric."""
    aggregate: dict[str, dict] = {}
    for metric_name, values in all_metrics.items():
        if values:
            aggregate[metric_name] = {
                "mean": round(statistics.mean(values), 4),
                "median": round(statistics.median(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "std_dev": round(statistics.stdev(values), 4) if len(values) > 1 else 0,
            }
    return aggregate
