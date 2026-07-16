#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR"/../../../../.env

INDEX_CONTENT_NDJSON_FILE_PATH=""
INDEX_SCHEMA_JSON_FILE_PATH=""
INDEX_NAME=""
OS_URL=$TEST_GEN_OPENSEARCH_URL
OS_USER=$TEST_GEN_OPENSEARCH_USERNAME
OS_PW=$TEST_GEN_OPENSEARCH_PASSWORD


print_usage() {
    printf "Usage: ./index_catalog.sh -n [index_name] -i [file-path-to-ndjson containing index content] -s [file-path-to-schema] [-h for usage] \n"
}

while getopts 'n:i:s:h' flag; do
    case "${flag}" in
        n) INDEX_NAME="${OPTARG}" ;;
        i) INDEX_CONTENT_NDJSON_FILE_PATH="${OPTARG}" ;;
        s) INDEX_SCHEMA_JSON_FILE_PATH="${OPTARG}" ;;
        h) print_usage
           exit 0 ;;
        *) print_usage
           exit 1 ;;
    esac
done

if [ -z "${INDEX_NAME}" ] || [ -z "${INDEX_CONTENT_NDJSON_FILE_PATH}" ]; then
    echo "No index name ('${INDEX_NAME}') or index content file ('${INDEX_CONTENT_NDJSON_FILE_PATH}') specified"
    print_usage
    exit 1
fi

if [ -n "${INDEX_SCHEMA_JSON_FILE_PATH}" ]; then
    echo -e "${MAJOR}Creating index, defining its mapping & settings\n${RESET}"
    curl -s -X PUT --insecure -u "$OS_USER":"$OS_PW" "$OS_URL/${INDEX_NAME}" -H 'Content-Type: application/json' --data-binary @"${INDEX_SCHEMA_JSON_FILE_PATH}"
    echo -e "\n"
fi

echo -e "${MAJOR}Indexing the data, please wait...\n${RESET}"
# Define the OpenSearch endpoint and content header
OPENSEARCH_URL="$OS_URL/${INDEX_NAME}/_bulk?pretty=false&filter_path=-items"
CONTENT_TYPE="Content-Type: application/x-ndjson"

# Using pre-prepared shrunk sample data for faster indexing
echo "Processing ${INDEX_CONTENT_NDJSON_FILE_PATH}"

# Send the file to OpenSearch using curl
curl -X POST --insecure -u "$OS_USER":"$OS_PW" "$OPENSEARCH_URL" -H "$CONTENT_TYPE" --data-binary @"${INDEX_CONTENT_NDJSON_FILE_PATH}"

# Check the response code to see if the request was successful
if [[ $? -ne 0 ]]; then
    echo "Failed to send sample data file"
else
    echo "Sample data file successfully sent to OpenSearch"
fi
