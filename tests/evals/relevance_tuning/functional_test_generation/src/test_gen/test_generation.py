import os
import sys
from dotenv import load_dotenv

# adding the src root to the path here so we can both call the file's main directly
# as well as call the script via src.test_gen.test_generation from direct parent dir
# of src package
script_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
sys.path.insert(0, f"{script_path}/../../")
# Put the project's own src/ ahead of site-packages so the top-level `tools`,
# `utils`, `agents` packages resolve to this project and not a same-named
# installed package (the agent server normally runs with cwd=src for this reason).
sys.path.insert(0, os.path.abspath(f"{script_path}/../../../../../../src"))
from src.test_gen.assertions import (
    CaseInsensitiveContainsAssertion,
    LLMRubricAssertion,
    TestCase,
    TestSuite,
)
from src.test_gen.cases.analysis import create_query_ctr_test_case, \
    create_document_ctr_test_case, create_top_queries_by_engagement_test_case, get_ubi_event_index_size_test_case, \
    get_worst_performing_queries_30_days_test_case, get_worst_performing_queries_test_case, \
    get_ask_for_query_improvement_asks_for_search_config_test_case
from src.test_gen.cases.configs import create_query_set_test_case, \
    create_search_configs_overview_test_case, create_query_set_by_id_test_case, create_query_set_w_queries_test_case, \
    create_search_config_w_search_fields_test_case
from src.test_gen.cases.experiment import \
    create_experiments_last_n_overview_test_case
from src.test_gen.cases.judgements import \
    create_single_judgement_list_test_case, create_judgement_list_overview_test_case
from src.test_gen.cases.model_cases import \
    create_investigate_query_improvement_by_boost_test_case, \
    create_investigate_query_improvement_by_spell_correction_test_case, get_samsung_query_improvement_test_case, \
    get_field_boost_offline_eval_test_case, get_search_config_copy_and_adjust_test_case, \
    get_multi_step_analysis_test_case_1
from src.test_gen.opensearch_client import get_client_manager
from src.utils.mcp_retrieval import init_mcp, shutdown


def build_test_suite_fast():
    return TestSuite(
        name="Fast Test Suite",
        description="Faster tests involving basic analysis / creation cases involving basic"
        " tool usage, yet without more involved analysis cases.",
        cases=tuple(
            [
                create_query_ctr_test_case(
                    "laptop", time_range_days=30, ubi_index="ubi_events"
                ),
                create_query_ctr_test_case(
                    "wireless earbuds", time_range_days=30, ubi_index="ubi_events"
                ),
                create_query_ctr_test_case(
                    "gold", time_range_days=30, ubi_index="ubi_events"
                ),
                create_document_ctr_test_case(
                    doc_id="B07ZCRSVBB", num_days=30, index="ubi_events"
                ),
                create_top_queries_by_engagement_test_case(
                    top_n=20,
                    min_search_volume=5,
                    time_range_days=30,
                    ubi_index="ubi_events",
                ),
                create_experiments_last_n_overview_test_case(
                    last_n=5
                ),
                create_judgement_list_overview_test_case(last_n=4),
                create_search_configs_overview_test_case(),
                create_query_set_by_id_test_case(),
                create_top_queries_by_engagement_test_case(
                    top_n=20,
                    min_search_volume=5,
                    time_range_days=30,
                    ubi_index="test_event_index",
                    query_index="test_query_index",
                ),
                # low ctr case
                create_document_ctr_test_case(
                    doc_id="shoe_low1", num_days=30, index="test_event_index"
                ),
                # high ctr case
                create_document_ctr_test_case(
                    doc_id="shoe_high1", num_days=30, index="test_event_index"
                ),
                # low ctr case
                create_document_ctr_test_case(
                    doc_id="trousers_green_low1", num_days=30, index="test_event_index"
                ),
                # high ctr case
                create_document_ctr_test_case(
                    doc_id="trousers_green_high1", num_days=30, index="test_event_index"
                ),
                get_ubi_event_index_size_test_case(),
                get_worst_performing_queries_30_days_test_case(),
                get_worst_performing_queries_test_case(),
                get_ask_for_query_improvement_asks_for_search_config_test_case(),
                create_single_judgement_list_test_case(),
                create_query_set_test_case(top_n=5)
            ]
        ),
    )


def build_test_suite_for_resource_generation():
    return TestSuite(
        name="Resource Generation Test Suite",
        description="Testing the creation of resources. Note that this alters"
                    " the state of available resources, and thus tests that check"
                    " available resources might need regeneration (or run these last)",
        cases=tuple(
            [
                # test cases which create resources
                # NOTE that these will alter available resources and thus
                # test cases that test the system state might need regeneration
                create_query_set_w_queries_test_case(),
                create_search_config_w_search_fields_test_case(),
            ]
        ),
    )


def build_test_suite_slow():
    return TestSuite(
        name="Slow Test Suite",
        description="More involved cases combining steps of analysis, hypothesis and validation.",
        cases=tuple(
            [
                create_investigate_query_improvement_by_boost_test_case(
                    query="shoe super",
                    doc_index="test_doc_index",
                    event_index="test_event_index",
                    boost_field="brand",
                ),
                create_investigate_query_improvement_by_boost_test_case(
                    query="sweet trousers green",
                    doc_index="test_doc_index",
                    event_index="test_event_index",
                    boost_field="color",
                ),
                create_investigate_query_improvement_by_spell_correction_test_case(
                    query="ninhendo swithc",
                    doc_index="test_doc_index",
                    event_index="test_event_index",
                    query_index="test_query_index",
                ),
                get_samsung_query_improvement_test_case(),
                get_field_boost_offline_eval_test_case(),
                get_search_config_copy_and_adjust_test_case(),
            ]
            + list(
                get_multi_step_analysis_test_case_1(
                    conversation_id="search_opt1_thread", run_id="run-1"
                )
            )
        ),
    )


def get_promptfoo_config_from_csv_cases(timeout: int = 300000) -> str:
    return f"""
description: "Config for tests from csv file"

providers:
  - id: 'file://../../../agent_request.py'
    label: 'Orchestrator invocation'
    config:
      timeout: {timeout}

defaultTest:
  options:
    provider:
      id: bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0
      config:
        region: us-east-1
        max_tokens: 1024

tests: file://tests.csv
    """


if __name__ == "__main__":
    # we are just calling here such that global object is set with the right credentials
    # NOTE that the client manager will have the url and credentials set
    # on the first call, all subsequent calls just reuse the same object
    # (if the global object is not cleared)
    load_dotenv(f"{script_path}/../../../../../../.env")
    # Direct client: still used for the search-relevance plugin reads (Category B).
    get_client_manager(
        opensearch_url=os.environ["TEST_GEN_OPENSEARCH_URL"],
        username=os.environ["TEST_GEN_OPENSEARCH_USERNAME"],
        password=os.environ["TEST_GEN_OPENSEARCH_PASSWORD"],
    )
    # MCP retrieval: UBI metric ground-truth goes through the agent's data path.
    init_mcp(
        username=os.environ["TEST_GEN_OPENSEARCH_USERNAME"],
        password=os.environ["TEST_GEN_OPENSEARCH_PASSWORD"],
    )
    test_suite_fast = build_test_suite_fast()
    test_suite_slow = build_test_suite_slow()
    test_suite_generation = build_test_suite_for_resource_generation()


    fast_test_path = f"{script_path}/../../../test_cases/functional/fast"
    slow_test_path = f"{script_path}/../../../test_cases/functional/slow"
    resource_generation_test_path = f"{script_path}/../../../test_cases/functional/generation"

    test_suite_fast.write(f"{fast_test_path}/tests.csv")
    with open(f"{fast_test_path}/eval.yaml", "w") as f:
        f.write(get_promptfoo_config_from_csv_cases(300000))
    test_suite_slow.write(f"{slow_test_path}/tests.csv")
    with open(f"{slow_test_path}/eval.yaml", "w") as f:
        f.write(get_promptfoo_config_from_csv_cases(3000000))
    test_suite_generation.write(f"{resource_generation_test_path}/tests.csv")
    with open(f"{resource_generation_test_path}/eval.yaml", "w") as f:
        f.write(get_promptfoo_config_from_csv_cases(300000))

    shutdown()
