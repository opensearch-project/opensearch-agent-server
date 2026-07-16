import asyncio
import json
from pydantic import BaseModel, ConfigDict
from src.test_gen.assertions import TestCase, LLMRubricAssertion
from src.test_gen.utils import Assertions
from src.utils.tool_utils import list_judgment_list, get_judgment


class JudgementRating(BaseModel):
    rating: str | float
    docId: str


class JudgementRatings(BaseModel):
    query: str
    ratings: list[JudgementRating]


class JudgementListResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    timestamp: str
    name: str
    status: str
    type: str
    metadata: dict[str, any]
    judgmentRatings: list[JudgementRatings]


def get_judgement_lists() -> list[JudgementListResults]:
    return [
        JudgementListResults(**x["_source"])
        for x in json.loads(asyncio.run(list_judgment_list()))["hits"]["hits"]
    ]


def get_judgement_list(judgement_id: str) -> JudgementListResults:
    return JudgementListResults(
        **json.loads(asyncio.run(get_judgment(judgment_id=judgement_id)))["hits"][
            "hits"
        ][0]["_source"]
    )


def get_most_recent_judgement_lists_llm_rubric_assertion(
    data: list[JudgementListResults],
) -> str:
    return Assertions.get_llm_rubric_assertion_yaml(
        data=data,
        relevant_fields=tuple(
            ["id", "timestamp", "name", "status", "type", "judgmentRatings"]
        ),
        field_key_to_description={},
        description_to_additional_values={},
    )


def create_judgement_list_overview_test_case(last_n: int) -> TestCase:
    results: list[JudgementListResults] = get_judgement_lists()
    results = list(reversed(sorted(results, key=lambda x: x.timestamp)))[:last_n]
    results = [
        x.model_copy(
            update={
                "judgmentRatings": [
                    y.model_copy(update={"ratings": y.ratings[:5]})
                    for y in x.judgmentRatings[:5]
                ]
            }
        )
        for x in results
    ]

    return TestCase(
        prompt=f"""
                Give me an overview of the last {last_n} most recently created judgement lists.
                Include the following attributes per experiment: the id, timestamp, name,
                status, type and judgement ratings. For the judgement ratings, give only examples
                for the first 5 queries and per query the first 5 example judgements.
                For those example judgements, explicitly give the document id and the score.
                """,
        assertions=tuple(
            [
                LLMRubricAssertion(
                    eval_prompt=get_most_recent_judgement_lists_llm_rubric_assertion(
                        data=results
                    )
                )
            ]
        ),
    )


def get_single_judgement_lists_llm_rubric_assertion(
    data: JudgementListResults,
) -> str:
    return Assertions.get_llm_rubric_assertion_yaml(
        data=[data],
        relevant_fields=tuple(
            ["id", "timestamp", "name", "status", "type", "judgmentRatings"]
        ),
        field_key_to_description={},
        description_to_additional_values={},
    )


def create_single_judgement_list_test_case() -> TestCase:
    all_judgement_lists: list[JudgementListResults] = (
        get_judgement_lists()
    )
    last_created_list: JudgementListResults = list(
        reversed(sorted(all_judgement_lists, key=lambda x: x.timestamp))
    )[0]
    result: JudgementListResults = get_judgement_list(
        last_created_list.id
    )
    result = result.model_copy(
        update={
            "judgmentRatings": [
                x.model_copy(update={"ratings": x.ratings[:5]})
                for x in result.judgmentRatings[:5]
            ]
        }
    )
    return TestCase(
        prompt="""
               Give me an overview of the most recently created judgement list.
               Include the following attributes per experiment: the id, timestamp, name,
               status, type and judgement ratings. For the judgement ratings, give only examples
               for the first 5 queries and per query the first 5 example judgements.
               For those example judgements, explicitly give the document id and the score.
               """,
        assertions=tuple(
            [
                LLMRubricAssertion(
                    eval_prompt=get_single_judgement_lists_llm_rubric_assertion(
                        data=result
                    )
                )
            ]
        ),
    )