#!/bin/bash

# script for passing agent name(s) and tests to run

set -eo pipefail

SCRIPT_DIR="$(dirname "$0")"
RUN_RELEVANCE_AGENT_FAST_TESTS=false
RUN_RELEVANCE_AGENT_SLOW_TESTS=false
RUN_RELEVANCE_AGENT_GENERATION_TESTS=false
RELEVANCE_AGENT_SCENARIO_NAME="scenario1"

print_usage() {
    printf "Usage: ./evaluate_agents.sh -t [comma-separated test names] [-h for usage] \n
    Usable test names so far: rta_fast, rta_slow, rta_generation, where rta_ prefix stands for relevance tuning agent."
}

while getopts 't:h' flag; do
    case "${flag}" in
        t) set -f
           IFS=,
           array=($OPTARG) ;;
        h) print_usage
           exit 0 ;;
        *) print_usage
           exit 1 ;;
    esac
done

run_relevance_agent_test_setup() {
  local scenario=$1
  # generate index data and test cases for programmatic scenarios
  echo "Generating index data for configured scenarios for relevance_tuning agent, scenario $scenario"
  "$SCRIPT_DIR"/../relevance_tuning/scripts/generate_index_data.sh -s $scenario

  echo "Indexing index data for configured scenarios for relevance_tuning agent, scenario $scenario"
  "$SCRIPT_DIR"/../relevance_tuning/scripts/create_all_indices.sh -s $scenario

  echo "Generating configured test cases for relevance_tuning agent"
  "$SCRIPT_DIR"/../relevance_tuning/scripts/generate_test_cases.sh
}

run_functional_relevance_agent_test(){
  test_suite_name=$1
  "$SCRIPT_DIR"/../relevance_tuning/scripts/run_eval.sh -k functional -t "$test_suite_name" -n
}

for i in "${array[@]}"; do
  case $i in
    rta_fast)
      RUN_RELEVANCE_AGENT_FAST_TESTS=true ;;
    rta_slow)
      RUN_RELEVANCE_AGENT_SLOW_TESTS=true ;;
    rta_generate)
      RUN_RELEVANCE_AGENT_GENERATION_TESTS=true ;;
  esac
done

# prepare the test setup
if [ $RUN_RELEVANCE_AGENT_FAST_TESTS ] || [ $RUN_RELEVANCE_AGENT_SLOW_TESTS ] || [ $RUN_RELEVANCE_AGENT_GENERATION_TESTS ] ; then
  run_relevance_agent_test_setup $RELEVANCE_AGENT_SCENARIO_NAME
fi

if [ $RUN_RELEVANCE_AGENT_FAST_TESTS ] ; then
  echo "Running fast evaluations for relevance_tuning agent"
  run_functional_relevance_agent_test fast
fi

if [ $RUN_RELEVANCE_AGENT_SLOW_TESTS ] ; then
  echo "Running slow evaluations for relevance_tuning agent"
  run_functional_relevance_agent_test slow
fi

if [ $RUN_RELEVANCE_AGENT_GENERATION_TESTS ] ; then
  echo "Running generation evaluations for relevance_tuning agent"
  run_functional_relevance_agent_test generation
fi
