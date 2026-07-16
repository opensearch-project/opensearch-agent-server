import asyncio
import json
from pydantic import BaseModel
from src.test_gen.formatting import FormattedModel
from src.test_gen.assertions import TestCase, LLMRubricAssertion
from src.test_gen.utils import Assertions
from src.utils.tool_utils import list_experiment


class ExperimentEvaluationResult(BaseModel):
    evaluationId: str | None = None
    query_text: str | None = None
    searchConfigurationId: str | None = None


class ErrorMsg(BaseModel):
    error: str


class ExperimentEvaluationSnapshot(BaseModel):
    searchConfigurationId: str
    docIds: list[str]


class Metric(FormattedModel):
    metric: str
    value: float


class ExperimentSnapshot(BaseModel):
    snapshots: list[ExperimentEvaluationSnapshot]
    metrics: list[Metric]
    query_text: str


class ListExperimentResult(BaseModel):
    timestamp: str
    type: str
    status: str
    querySetId: str
    searchConfigurationList: list[str]
    judgmentList: dict[str, str] | list[str]
    size: int
    isScheduled: bool
    scheduledExperimentJobId: str | None
    results: list[ExperimentEvaluationResult | ErrorMsg | ExperimentSnapshot]


def get_experiments() -> list[ListExperimentResult]:
    """
    Provide overview of all experiments.
    The experiment output is same as calling get_experiment(experiment_id) for all
    experimentIds.
    :return:
    """
    return [
        ListExperimentResult(**x["_source"])
        for x in json.loads(asyncio.run(list_experiment()))["hits"]["hits"]
    ]


def get_top_n_experiments_overview_llm_rubric_assertion(
    data: list[ListExperimentResult],
) -> str:
    return Assertions.get_llm_rubric_assertion_yaml(
        data=data,
        relevant_fields=tuple(
            [
                "type",
                "status",
                "querySetId",
                "searchConfigurationList",
                "judgmentList"
            ]
        ),
        field_key_to_description={},
        description_to_additional_values={},
    )


def create_experiments_last_n_overview_test_case(last_n: int) -> TestCase:
    experiments: list[ListExperimentResult] = get_experiments()
    experiments = list(reversed(sorted(experiments, key=lambda x: x.timestamp)))[
        :last_n
    ]

    return TestCase(
        prompt=f"""
                Give me an overview of the last {last_n} most recent experiments.
                Order the experiment from newest to oldest.
                Include for each experiment the type, status, the query set id, 
                the list of search configuration ids, the list of judgement list ids.
                Do not just give the count of ids in search configuration list or
                the judgement list list, make sure to include the ids.
                """,
        assertions=tuple(
            [
                LLMRubricAssertion(
                    eval_prompt=get_top_n_experiments_overview_llm_rubric_assertion(
                        data=experiments
                    )
                )
            ]
        ),
    )