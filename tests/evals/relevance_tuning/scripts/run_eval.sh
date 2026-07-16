#!/bin/bash

set -o pipefail

SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
source "$PROJECT_ROOT/.env"

# Point promptfoo's Python worker at the project venv so deps like `requests`
# are available. The venv is created on first `uv run` (generate_test_cases.sh).
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

TEST_FOLDER=''
OUTPUT_BASE_NAME=$(date +%s)
OUTPUT_FORMATS='json,csv,html'
OVERWRITE=false
TEST_TYPE='functional'
EVAL_OPTIONS=''


print_usage() {
    printf "Usage: ./run_eval.sh -k [TYPE of test, e.g 'functional' or 'redteam'] -t [TEST_SUITE_NAME, e.g single_turn] -b [OUTPUT_BASE_NAME, e.g the eval result filename without format suffix] -f [formats, comma separated if multiple, e.g html,xml,json,jsonl,csv] [-n for --no-cache] [-h for usage] [-d if overwrite]\n"
}

while getopts 'k:t:b:f:dn' flag; do
    case "${flag}" in
        k) TEST_TYPE="${OPTARG}" ;;
        t) TEST_FOLDER="${OPTARG}" ;;
        b) OUTPUT_BASE_NAME="${OPTARG}" ;;
        f) OUTPUT_FORMATS="${OPTARG}" ;;
        d) OVERWRITE=true ;;
        n) EVAL_OPTIONS="--no-cache" ;;
        h) print_usage
           exit 0 ;;
        *) print_usage
           exit 1 ;;
    esac
done

EVAL_CONFIG_FILE="eval.yaml"
EVAL_CONFIG_SUBFOLDER="functional"
echo "$TEST_TYPE" | grep -q "functional"
IS_FUNCTIONAL=$?
echo "$TEST_TYPE" | grep -q "redteam"
IS_REDTEAM=$?
if [ $IS_FUNCTIONAL -eq 1 ] && [ $IS_REDTEAM -eq 1 ]; then
    echo "Passed test type is not one of ['functional', 'redteam'], but '${TEST_TYPE}'. Exiting"
    exit 1
fi
if [ $IS_REDTEAM -eq 0 ]; then
    EVAL_CONFIG_SUBFOLDER="redteam"
fi

EVAL_CONFIG_FILE="$SCRIPT_DIR/../test_cases/$EVAL_CONFIG_SUBFOLDER/$TEST_FOLDER/$EVAL_CONFIG_FILE"
OUTPUT_BASE="$SCRIPT_DIR/../test_cases/$EVAL_CONFIG_SUBFOLDER/$TEST_FOLDER/results/$OUTPUT_BASE_NAME"

if [ ! -e "${EVAL_CONFIG_FILE}" ]; then
    echo "Chosen eval config file '${EVAL_CONFIG_FILE}' does not exist. Exiting."
    exit 1
fi

run_promptfoo(){
    IFS=',' read -r -a output_format_array <<< "$OUTPUT_FORMATS"
    OUTPUT_STRING=''
    for format_index in "${!output_format_array[@]}"
    do
      ADD_OUTPUT="${OUTPUT_BASE}.${output_format_array[format_index]}"
      if [ -e "$ADD_OUTPUT" ] && [ "$OVERWRITE" = false ]; then
        echo "$ADD_OUTPUT already exists and overwrite = false, skipping processing."
        exit 1
      fi
      OUTPUT_STRING="$OUTPUT_STRING --output $ADD_OUTPUT"
    done
    USED_CMD="PROMPTFOO_PYTHON=$VENV_PYTHON AWS_BEARER_TOKEN_BEDROCK=$AWS_BEARER_TOKEN_BEDROCK promptfoo eval -c $EVAL_CONFIG_FILE $OUTPUT_STRING $EVAL_OPTIONS"
    eval $USED_CMD
}

run_promptfoo


