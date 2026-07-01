import asyncio
import json
from pydantic import BaseModel
from src.test_gen.assertions import TestCase, LLMRubricAssertion
from src.test_gen.utils import Assertions
from src.utils.tool_utils import list_query_set, get_query_set, \
    list_search_configuration


class QuerySetQuery(BaseModel):
    queryText: str


class QuerySetResults(BaseModel):
    id: str
    name: str
    description: str
    sampling: str
    timestamp: str
    querySetQueries: list[QuerySetQuery]


class SearchConfiguration(BaseModel):
    id: str
    name: str
    timestamp: str
    index: str
    query: str
    searchPipeline: str


def get_query_sets() -> list[QuerySetResults]:
    return [
        QuerySetResults(**x["_source"])
        for x in json.loads(asyncio.run(list_query_set()))["hits"]["hits"]
    ]


def get_query_set_by_id(id: str) -> QuerySetResults:
    return QuerySetResults(
        **json.loads(asyncio.run(get_query_set(id)))["hits"]["hits"][0]["_source"]
    )


def get_all_search_configs() -> list[SearchConfiguration]:
    return [
        SearchConfiguration(**x["_source"])
        for x in json.loads(asyncio.run(list_search_configuration()))["hits"][
            "hits"
        ]
    ]


def get_query_sets_llm_rubric_assertion(data: list[QuerySetResults]) -> str:
    return Assertions.get_llm_rubric_assertion_yaml(
        data=data,
        relevant_fields=tuple(
            ["id", "name", "descriptionsampling", "timestamp", "querySetQueries"]
        ),
        field_key_to_description={},
        description_to_additional_values={},
        extra_notes="When comparing the expected query set with the result, small "
        "query sets with <= 10 elements should be listed fully in the "
        "response and reflect the expectation, while for longer lists a "
        "subsample of 10 queries is sufficient. Make sure all of those "
        "queries also appear in the listed expectation.",
    )


def create_query_set_test_case(top_n: int,
                               max_num_queries_per_set: int = 10) -> TestCase:
    result: list[QuerySetResults] = [x.model_copy(update={"querySetQueries": x.querySetQueries[:max_num_queries_per_set]}) for x in get_query_sets()]
    # Prompt asks for the last top_n most recently created; slice to match.
    result = list(reversed(sorted(result, key=lambda x: x.timestamp)))[:top_n]

    return TestCase(
        prompt=f"""
               Give me an overview of the last {top_n} most recently created query sets.
               Include the following attributes per query set: the id, name,
               description, timestamp and the actual query set, listing a sample
               of {max_num_queries_per_set} example queries contained in the respective query set.
               """,
        assertions=tuple(
            [
                LLMRubricAssertion(
                    eval_prompt=get_query_sets_llm_rubric_assertion(
                        data=result
                    )
                )
            ]
        ),
    )


def create_query_set_by_id_test_case() -> TestCase:
    available_configs = get_query_sets()
    available_configs = list(
        reversed(sorted(available_configs, key=lambda x: x.timestamp))
    )[0]
    result: QuerySetResults = get_query_set_by_id(
        available_configs.id
    )

    return TestCase(
        prompt=f"""
               Give me an overview of the query set with id '{result.id}'.
               Include the following attributes per query set: the id, name,
               description, timestamp and the actual query set. If the query set
               contains <= 10 elements, list the queries fully, otherwise
               list 10 example queries.
               """,
        assertions=tuple(
            [
                LLMRubricAssertion(
                    eval_prompt=get_query_sets_llm_rubric_assertion(
                        data=[result]
                    )
                )
            ]
        ),
    )


def get_search_configs_llm_rubric_assertion(data: list[SearchConfiguration]) -> str:
    return Assertions.get_llm_rubric_assertion_yaml(
        data=data,
        relevant_fields=tuple(
            ["id", "name", "index", "query", "searchPipeline"]
        ),
        field_key_to_description={},
        description_to_additional_values={},
    )


def create_search_configs_overview_test_case() -> TestCase:
    search_configs: list[SearchConfiguration] = (
        get_all_search_configs()
    )[:10]
    return TestCase(
        prompt="""
               Give me an overview of the available search configurations.
               If less than ten configurations are available, list all, otherwise
               list the first 10 you find.
               Include the following attributes per query set: the id (the actual search configuration id), name, index, query and search pipeline (if any).
               As per the query, do give me the full query dsl and a description of what the query does.
               """,
        assertions=tuple(
            [
                LLMRubricAssertion(
                    eval_prompt=get_search_configs_llm_rubric_assertion(
                        data=search_configs
                    )
                )
            ]
        ),
    )


def create_query_set_w_queries_test_case() -> TestCase:
    return TestCase(
        prompt="""
        Create a new query set with the following queries:
          laptop stand
          wireless earbuds
          hoodie
          wine glass
          rc cars
          toys for boys
          lego friends
          othoking bamboo charging station
          phone charger
          alarm clock
          karaoke microphone
          samsung
        """,
        assertions=[
            LLMRubricAssertion(
                eval_prompt="""
                A new query set was created with the 12 queries mentioned in the prompt.
                """
            )
        ],
    )


def create_search_config_w_search_fields_test_case() -> TestCase:
    return TestCase(
        prompt="""
          Create a search configuration that searches across the fields
          asin, title, category, bullet_points, description, brand, color with a multi_match
          query and call it my_baseline_2. It searches the ecommerce index.
            """,
        assertions=[
            LLMRubricAssertion(
                eval_prompt="""
                One new search configuration with the name 'my_baseline_2' and a multi_match
                query across the mentioned fields targeting the ecommerce index.
                """
            )
        ],
    )