import asyncio
import json
from typing import Collection
from src.test_gen.assertions import TestCase, CaseInsensitiveContainsAssertion
from src.test_gen.assertions import LLMRubricAssertion
from src.utils.tool_utils import (
    get_document_ctr,
    get_judgment,
    get_query_ctr,
    get_query_performance_metrics,
    get_query_set,
    get_top_queries_by_engagement,
    list_experiment,
    list_judgment_list,
    list_query_set,
    list_search_configuration,
)
from src.test_gen.utils import Assertions
from src.utils.utils import flatten_lists
from src.test_gen.formatting import FormattedModel
from src.test_gen.formatting import format_value


class ClickResult(FormattedModel):
    query_text: str
    search_volume: int
    searches_with_clicks: int
    total_clicks: int
    ctr_percentage: float
    average_clicks_per_search: float
    zero_click_rate_percentage: float


class DocumentCTR(FormattedModel):
    document_id: str
    # None when the prompt states no time window (metrics over all data).
    time_range_days: int | None = None
    total_impressions: int
    total_clicks: int
    ctr_percentage: float
    average_position_when_clicked: float | None



def get_worst_queries(
    zero_ctr_n: int = 8,
    high_volume_n: int = 3,
    time_range_days: int | None = None,
    ubi_index: str = "ubi_events",
) -> dict[str, list[ClickResult]]:
    """Worst-performing queries pulled the SAME way the agent pulls them (via the
    MCP-backed metric tool), split into the TWO groups the analysis should surface:

      - "zero_ctr": queries nobody clicks (0% CTR), worst first by search volume —
        a busy query with no clicks is the worst performer.
      - "high_volume_low_ctr": the highest-VOLUME queries that still have a nonzero
        but sub-optimal CTR (biggest improvement opportunities / potential high
        impact). Derived by ranking on volume, NOT by lowest CTR — a pure lowest-CTR
        sort only yields zero-click queries and hides this group.

    Recomputed at generation time so both groups always track the CURRENT index
    instead of hard-coded query strings that drift as the data changes.
    """
    # top_n huge => the tool returns every query; we re-rank into the two groups.
    all_metrics = json.loads(
        asyncio.run(
            get_query_performance_metrics(
                query_text=None,
                top_n=100_000,
                time_range_days=time_range_days,
                ubi_index=ubi_index,
            )
        )
    )
    # Drop anomalous rows with no real searches (0 volume but stray clicks) and
    # blank query texts (nothing nameable for the assertion to match on).
    real = [
        ClickResult(**x) for x in all_metrics["queries"]
        if x["search_volume"] > 0 and (x["query_text"] or "").strip()
    ]
    zero_ctr = sorted(
        (q for q in real if q.ctr_percentage == 0),
        key=lambda q: -q.search_volume,
    )
    high_volume_low_ctr = sorted(
        (q for q in real if q.ctr_percentage > 0),
        key=lambda q: (-q.search_volume, q.ctr_percentage),
    )
    return {
        "zero_ctr": zero_ctr[:zero_ctr_n],
        "high_volume_low_ctr": high_volume_low_ctr[:high_volume_n],
    }


def get_worst_queries_rubric(
    zero_ctr: Collection[ClickResult],
    high_volume_low_ctr: Collection[ClickResult],
    min_zero_ctr: int,
) -> str:
    zero_list = "\n".join(
        f"- {q.query_text} ({q.search_volume} searches, 0 clicks)" for q in zero_ctr
    )
    high_vol_list = "\n".join(
        f"- {q.query_text} ({q.search_volume} searches, CTR {q.ctr_percentage:.2f}%)"
        for q in high_volume_low_ctr
    )
    return f"""
        The user asked for the worst performing queries. Using the same
        user-behaviour data the agent has access to, the worst performers for the
        CURRENT index fall into two groups.

        Group A - zero click-through (nobody clicks these), worst first by volume:
        {zero_list}

        Group B - highest-VOLUME queries with a sub-optimal (nonzero) CTR, i.e. the
        biggest improvement opportunities / potential high impact:
        {high_vol_list}

        Pass the assertion only if ALL of the following hold:
        1) The response identifies poorly / worst performing queries and names AT
           LEAST {min_zero_ctr} of the Group A queries above (match on the query
           text; treat obvious typos or minor variants as matches).
        2) The response ALSO calls out AT LEAST ONE of the Group B high-volume
           queries above as a high-volume / high-impact query with room to improve
           its CTR.
        3) The response suggests next steps that include at least one of: generating
           hypotheses, or analyzing the search results for the problematic queries.

        Ignore ordering, grouping labels, exact CTR/volume numbers, and any extra
        queries the response mentions beyond these lists.
        """


def get_worst_performing_queries_test_case():
    # Ground truth pulled live from the index (not hard-coded) and matched loosely,
    # but STILL covering both groups: zero-CTR queries AND high-volume low-CTR
    # queries, so the assertion stays valid as the underlying data changes.
    worst = get_worst_queries(time_range_days=None)
    zero_ctr = worst["zero_ctr"]
    high_volume_low_ctr = worst["high_volume_low_ctr"]
    min_zero_ctr = min(3, len(zero_ctr))
    all_texts = [q.query_text for q in zero_ctr] + [
        q.query_text for q in high_volume_low_ctr
    ]
    return TestCase(
        prompt="Find the 10 worst performing queries.",
        assertions=[
            LLMRubricAssertion(
                eval_prompt=get_worst_queries_rubric(
                    zero_ctr, high_volume_low_ctr, min_zero_ctr
                )
            ),
            # Cheap sanity net: at least one of the computed worst queries appears.
            CaseInsensitiveContainsAssertion(
                contains_all=False,
                contains_texts=all_texts,
            ),
        ],
    )


def get_ubi_event_index_size_test_case():
    return TestCase(
        prompt="""
            How many events are there in the ubi_events index? 
            Go ahead with the analysis and directly come back to me with an answer.
            """,
        assertions=[CaseInsensitiveContainsAssertion(contains_all=True, contains_texts=["3,448"])],
    )

def get_worst_performing_queries_30_days_test_case():
    return TestCase(
        prompt="""
        What are the worst performing queries of the past 30 days? 
        Go ahead with the analysis and directly come back to me with an answer.
        """,
        assertions=[
            CaseInsensitiveContainsAssertion(
                contains_all=False, contains_texts=[
                    "spiderwire stealth",
                    "wireless earbuds",
                    "gold",
                    "wireless",
                    "wirel",
                    "wirele",
                    "wirelese",
                    "wirels"
                ]
            )
        ],
    )

def get_ask_for_query_improvement_asks_for_search_config_test_case():
    return TestCase(
        prompt="""
        Help me improve the query samsung under the current search setup.
        """,
        assertions=[
            LLMRubricAssertion(
                eval_prompt="""
                Response contains an ask for the search configuration or plausible suggestions
                based on observations from tracking data and / or reference ot the hypothesis agent.
                """
            )
        ],
    )


def get_click_results_top_n(
    top_n: int = 20, time_range_days: int = 30, ubi_index: str = "ubi_events"
) -> Collection[ClickResult]:
    click_coroutine = get_query_performance_metrics(
        query_text=None,
        top_n=top_n,
        time_range_days=time_range_days,
        ubi_index=ubi_index,
    )
    click_result = json.loads(asyncio.run(click_coroutine))
    return [ClickResult(**x) for x in click_result["queries"]]


def get_top_n_ctr_llm_rubric_assertion(
    data: Collection[ClickResult],
    total_queries_analyzed: int
) -> str:
    return Assertions.get_llm_rubric_assertion_yaml(
        data=data,
        relevant_fields=tuple(
            [
                "query_text",
                "search_volume",
                "searches_with_clicks",
                "total_clicks",
                "ctr_percentage",
                "average_clicks_per_search",
                "zero_click_rate_percentage",
            ]
        ),
        field_key_to_description={
            "query_text": "the user query",
            "search_volume": "total number of searches",
            "searches_with_clicks": "number of searches with clicks",
            "total_clicks": "total number of clicks",
            "ctr_percentage": "CTR in percentage, rounded to two decimals",
            "average_clicks_per_search": "average number of clicks per search for the given query",
            "zero_click_rate_percentage": "percentage of searches for the given query for which no click occurred",
        },
        description_to_additional_values={
            "Total number of queries contained in the result": total_queries_analyzed
        },
    )


def create_top_n_ctr_test_case(
    top_n: int = 20,
    time_range_days: int = 30,
    ubi_index: str = "ubi_events"
) -> TestCase:
    tool_results: Collection[ClickResult] = get_click_results_top_n(
        top_n=top_n,
        time_range_days=time_range_days,
        ubi_index=ubi_index
    )
    return TestCase(
        prompt=f"""Analyze the performance of the top {top_n} results in terms of CTR. 
        For each query, determine the following properties: 
        total query volume, searches with clicks and total number of clicks. 
        Give the average clicks per search, the zero click rate and ctr. 
        The latter two give in percentages formatted with two decimals.""",
        assertions=tuple(
            [
                CaseInsensitiveContainsAssertion(
                    contains_all=True,
                    contains_texts=flatten_lists(
                        [
                            list(x.print_representation().values())
                            for x in tool_results
                        ]
                    ),
                ),
                LLMRubricAssertion(
                    eval_prompt=get_top_n_ctr_llm_rubric_assertion(
                        tool_results,
                        total_queries_analyzed=top_n
                    )
                ),
            ]
        ),
    )


def get_query_ctr_rubric_assertion(data: ClickResult) -> str:
    return Assertions.get_llm_rubric_assertion_yaml(
        data=[data],
        relevant_fields=tuple(
            [
                "query_text",
                "search_volume",
                "searches_with_clicks",
                "total_clicks",
                "ctr_percentage",
                "average_clicks_per_search",
                "zero_click_rate_percentage",
            ]
        ),
        field_key_to_description={
            "query_text": "the user query",
            "search_volume": "total number of searches",
            "searches_with_clicks": "number of searches with clicks",
            "total_clicks": "total number of clicks",
            "ctr_percentage": "CTR in percentage, rounded to two decimals",
            "average_clicks_per_search": "average number of clicks per search for the given query",
            "zero_click_rate_percentage": "percentage of searches for the given query for which no click occurred",
        },
        description_to_additional_values={},
    )


def create_query_ctr_test_case(
    query: str, time_range_days: int = 30, ubi_index: str = "ubi_events"
) -> TestCase:
    query_ctr_result_json = json.loads(
        asyncio.run(
            get_query_ctr(
                query,
                # This prompt states no time window, so compute over all data
                # (matching the agent's behaviour for an unwindowed request).
                time_range_days=None,
                ubi_index=ubi_index,
            )
        )
    )
    query_ctr_result_json["search_volume"] = query_ctr_result_json["total_searches"]
    query_ctr_result: ClickResult = ClickResult(**query_ctr_result_json)

    return TestCase(
        prompt=f"""Analyze the performance of the query '{query}'. 
        Specifically, for the given query, determine and return the following
        properties: total query volume, searches with clicks and total number of clicks. 
        Further return the average clicks per search, the zero click rate and ctr. 
        The latter two give in percentages formatted with two decimals.
        """,
        assertions=tuple(
            [
                CaseInsensitiveContainsAssertion(
                    contains_all=True,
                    contains_texts=tuple(
                        [
                            format_value(query_ctr_result.search_volume),
                            format_value(query_ctr_result.searches_with_clicks),
                            format_value(query_ctr_result.total_clicks),
                            format_value(query_ctr_result.ctr_percentage),
                            format_value(query_ctr_result.average_clicks_per_search),
                            format_value(query_ctr_result.zero_click_rate_percentage)
                        ]
                    ),
                ),
                LLMRubricAssertion(
                    eval_prompt=get_query_ctr_rubric_assertion(
                        query_ctr_result
                    )
                ),
            ]
        ),
    )


def get_document_ctr_rubric_assertion(data: DocumentCTR) -> str:
    return Assertions.get_llm_rubric_assertion_yaml(
        data=[data],
        relevant_fields=tuple(
            [
                "document_id",
                "total_impressions",
                "total_clicks",
                "ctr_percentage",
                "average_position_when_clicked",
            ]
        ),
        field_key_to_description={
            "document_id": "id of the document",
            "total_impressions": "number of times users saw the document",
            "total_clicks": "number of times the document was clicked",
            "ctr_percentage": "CTR in percentage, rounded to two decimals",
            "average_position_when_clicked": "average position in the search result for clicks on the document",
        },
        description_to_additional_values={},
    )


def create_document_ctr_test_case(
    doc_id: str, num_days: int = 30, index: str = "ubi_events"
) -> TestCase:
    result = json.loads(
        asyncio.run(
            get_document_ctr(
                doc_id=doc_id,
                # No time window in this prompt — compute over all data.
                time_range_days=None,
                ubi_index=index,
            )
        )
    )
    result = DocumentCTR(**result)
    contains_texts = tuple(
        [
            format_value(result.total_impressions),
            format_value(result.total_clicks),
            format_value(result.ctr_percentage, 2),
        ]
    )
    if result.average_position_when_clicked:
        contains_texts = tuple(
            list(contains_texts)
            + [format_value(result.average_position_when_clicked)]
        )

    return TestCase(
        prompt=f"""Analyze the performance of the product with document_id '{doc_id}'
            in the index '{index}'.
            For the given document, provide the total number of impressions, the
            total number of clicks, ctr in percentage (rounded and formatted to 2 decimals)
            and the average click position.""",
        assertions=tuple(
            [
                CaseInsensitiveContainsAssertion(contains_all=True, contains_texts=contains_texts),
                LLMRubricAssertion(
                    eval_prompt=get_document_ctr_rubric_assertion(
                        result
                    )
                ),
            ]
        ),
    )


def create_top_queries_by_engagement_test_case(
    top_n: int = 20,
    min_search_volume: int = 5,
    time_range_days: int = 30,
    ubi_index: str = "ubi_events",
    query_index: str | None = None,
) -> TestCase:
    result = asyncio.run(
        get_top_queries_by_engagement(
            top_n=top_n,
            min_search_volume=min_search_volume,
            # Prompt states "over a time range of {time_range_days} days" —
            # filter to match so ground-truth aligns with the agent.
            time_range_days=time_range_days,
            ubi_index=ubi_index,
            query_index=query_index,
        )
    )
    result = json.loads(result)
    total_queries_analyzed: int = result["total_queries_analyzed"]
    result = [ClickResult(**x) for x in result["queries"]]
    # Search volume is pulled from the query table; clicks from the event table.
    query_table = query_index or ("ubi_queries" if ubi_index == "ubi_events" else ubi_index)
    return TestCase(
        prompt=f"""Analyze the top {top_n} queries performance by engagement.
                Take the search/query volume from the query index '{query_table}'
                and the click (event) counts from the event index '{ubi_index}'.
                Only consider queries with at least
                {min_search_volume} searches, and over a time range of {time_range_days}
                days. Determine the following properties per query:
                total query volume, searches with clicks and total number of clicks,
                average clicks per search, the zero click rate and ctr.
                The latter two give in percentages formatted with two decimals.
                Also state how many queries matched the above filter criteria.
                Do not use grouping delimiter for amounts over one thousand, only use decimal
                delimiter where applicable.
                """,
        assertions=tuple(
            [
                CaseInsensitiveContainsAssertion(
                    contains_all=True,
                    contains_texts=flatten_lists(
                        [list(x.print_representation().values()) for x in result]
                    )
                    + [str(total_queries_analyzed)],
                ),
                LLMRubricAssertion(
                    eval_prompt=get_top_n_ctr_llm_rubric_assertion(
                        data=result,
                        total_queries_analyzed=total_queries_analyzed
                    )
                ),
            ]
        ),
    )