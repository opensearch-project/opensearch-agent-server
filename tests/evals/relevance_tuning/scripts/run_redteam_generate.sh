#!/bin/bash

set -o pipefail

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR"/../../../../.env

TEST_FOLDER=''
OVERWRITE=false

INPUT_CONFIG_FILE_NAME="promptfooconfig_redteam.yaml" # needs to refer to file in tests_promptfoo root

print_usage() {
    printf "Usage: ./run_redteam_generate.sh -t [TEST_SUITE_NAME, e.g medium_german] [-c input_config, default is promptfooconfig_redteam.yaml in root test folder] [-h for usage] [-d if overwrite]\n"
}

while getopts 't:c:dh' flag; do
    case "${flag}" in
        t) TEST_FOLDER="${OPTARG}" ;;
        c) INPUT_CONFIG="${OPTARG}" ;;
        d) OVERWRITE=true ;;
        h) print_usage
           exit 0 ;;
        *) print_usage
           exit 1 ;;
    esac
done

INPUT_CONFIG="$SCRIPT_DIR/../${INPUT_CONFIG_FILE_NAME}"
TARGET_EVAL_CONFIG_FILE="eval.yaml"
TARGET_EVAL_PATH="$SCRIPT_DIR/../test_cases/redteam/$TEST_FOLDER"
TARGET_EVAL_CONFIG_FILE="$TARGET_EVAL_PATH/$TARGET_EVAL_CONFIG_FILE"
TARGET_INPUT_CONFIG_COPY_FILE="$TARGET_EVAL_PATH/promptfooconfig.yaml"

if [ -e "${TARGET_EVAL_CONFIG_FILE}" ] && [ "$OVERWRITE" = false ]; then
    echo "Chosen target eval config file '${TARGET_EVAL_CONFIG_FILE}' already exists and overwrite = false, skipping processing."
    exit 1
fi

run_promptfoo_generate(){
    USED_CMD="AWS_BEARER_TOKEN_BEDROCK=$AWS_BEARER_TOKEN_BEDROCK promptfoo redteam generate -c $INPUT_CONFIG -o $TARGET_EVAL_CONFIG_FILE"
    eval "$USED_CMD"
}

mkdir -p "$TARGET_EVAL_PATH"
run_promptfoo_generate
# copy the promptfoo config used to generate
cp "$INPUT_CONFIG" "$TARGET_INPUT_CONFIG_COPY_FILE"


