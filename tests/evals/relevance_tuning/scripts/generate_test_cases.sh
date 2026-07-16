#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(dirname "$0")"
CWD=$(pwd)

cd "$SCRIPT_DIR"/../functional_test_generation
uv run --extra evals python -m src.test_gen.test_generation
cd "$CWD"
