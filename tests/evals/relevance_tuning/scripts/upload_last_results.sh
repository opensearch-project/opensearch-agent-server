#!/bin/bash

set -o pipefail

# NOTE: for the above to work, u will need to set the respective env vars that are suitable for basic auth
# export PROMPTFOO_SHARE_USERNAME=<username>
# export PROMPTFOO_SHARE_PASSWORD=<pw>
PROMPTFOO_URL="http://$PROMPTFOO_SHARE_USERNAME:$PROMPTFOO_SHARE_PASSWORD@chorus-opensearch-edition.dev.o19s.com:8000"
PROMPTFOO_REMOTE_API_BASE_URL=$PROMPTFOO_URL PROMPTFOO_REMOTE_APP_BASE_URL=$PROMPTFOO_URL promptfoo share
