#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(dirname "$0")"

SCENARIO="scenario1"

print_usage() {
    printf "Usage: ./generate_index_data.sh -s [scenario_name] [-h for usage] \n"
}

while getopts 's:h' flag; do
    case "${flag}" in
        s) SCENARIO="${OPTARG}" ;;
        h) print_usage
           exit 0 ;;
        *) print_usage
           exit 1 ;;
    esac
done



uv run --extra evals python "$SCRIPT_DIR"/../functional_test_generation/src/index_gen/index_scenarios.py --scenario_name "$SCENARIO"
