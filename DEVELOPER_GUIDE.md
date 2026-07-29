![OpenSearch logo](https://github.com/opensearch-project/opensearch-py/raw/main/OpenSearch.svg)

# OpenSearch Agent Server Developer Guide

## Table of Contents
- [Overview](#overview)
- [Development Setup](#development-setup)
- [Running the Server](#running-the-server)
- [Managing Dependencies](#managing-dependencies)
- [Adding a New Agent](#adding-a-new-agent)
- [Code Style](#code-style)
- [Testing](#testing)

## Overview

This guide is for developers who want to contribute to the OpenSearch Agent Server project. It covers local development setup, running the server, and how to add new agents.

The server is a multi-agent orchestrator that routes requests from OpenSearch Dashboards to specialized sub-agents based on page context. It exposes an [AG-UI](https://github.com/ag-ui-protocol/ag-ui) compatible HTTP API and uses [Strands Agents](https://github.com/strands-agents/sdk-python) as the agent runtime.

## Development Setup

### 1. Clone the Repository

```bash
git clone git@github.com:opensearch-project/opensearch-agent-server.git
cd opensearch-agent-server
```

### 2. Install Dependencies

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
# Create venv and install all dependencies (including dev extras)
uv sync --all-extras

# Activate the venv
source .venv/bin/activate
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` to set your OpenSearch URL, credentials, and LLM provider. At minimum you need:

- `OPENSEARCH_URL` — your OpenSearch cluster endpoint
- One LLM provider configured (AWS Bedrock or Ollama — see `.env.example` for options)
- `MCP_SERVER_URL` — URL of a running [opensearch-mcp-server-py](https://github.com/opensearch-project/opensearch-mcp-server-py) instance

### After Changing Dependencies

```bash
# After editing pyproject.toml, update lock file and sync
uv lock
uv sync --all-extras
```

## Running the Server

```bash
# With the venv activated
source .venv/bin/activate
opensearch-agent-server

# Or without activating
uv run opensearch-agent-server
```

The server starts on `http://localhost:8001` by default. OpenSearch Dashboards connects to it via the AG-UI protocol.

## Managing Dependencies

```bash
# Add a runtime dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Update all dependencies to latest allowed versions
uv lock --upgrade
uv sync --all-extras
```

## Adding a New Agent

Follow the pattern established by `src/agents/art/`. Registration happens in `src/server/ag_ui_app.py`.

### 1. Create your agent directory

Add a subdirectory under `src/agents/` with at minimum an `__init__.py` and a module containing your factory function:

```
src/agents/my_agent/
    __init__.py
    my_agent.py
    specialized_agents.py  # optional: constituent sub-agents
```

In `my_agent.py`, implement a factory function that returns a configured [Strands](https://github.com/strands-agents/sdk-python) `Agent`:

```python
from strands import Agent

def create_my_agent(opensearch_url: str) -> Agent:
    return Agent(
        system_prompt="You are an OpenSearch assistant ...",
        tools=[...],
    )
```

### 2. Register the agent in `ag_ui_app.py`

In `src/server/ag_ui_app.py`, import your factory and add two registrations alongside the existing `art` and `default` entries:

```python
from agents.my_agent.my_agent import create_my_agent
from orchestrator.registry import AgentRegistration

# 1. Register with the AgentRegistry (used for routing)
registry.register(AgentRegistration(
    name="my-agent",
    description="What this agent does.",
    page_contexts=["my-page-context"],  # Dashboards page(s) that route here
    is_default=False,
))

# 2. Register the factory with the orchestrator (used to instantiate per request)
orchestrator.register_agent_factory(
    name="my-agent",
    factory=lambda: create_my_agent(opensearch_url),
    description="What this agent does.",
    config=context_config,
)
```

The `page_contexts` list controls which OpenSearch Dashboards page context strings route to your agent. Any unmatched context falls through to the default agent.

### 3. Write tests

Add unit tests under `tests/` covering your factory function and any agent-specific logic. See existing tests for patterns.

### 4. Update documentation

If your agent adds new environment variables or configuration options, update `README.md` and `.env.example`.

## Code Style

This project is configured to use [ruff](https://docs.astral.sh/ruff/) for linting and formatting (see `[tool.ruff]` in `pyproject.toml`). Running it before opening a PR is recommended:

```bash
# Format code
uv run ruff format .

# Check for lint issues
uv run ruff check .
```

All new source files must include the Apache 2.0 license header:

```python
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run only unit tests
uv run pytest -m unit

# Run with coverage report
uv run pytest --cov=server --cov=agents --cov=orchestrator
```

### Integration Tests

```bash
uv run pytest -m integration
```

All new features and bug fixes must include tests. PRs that reduce test coverage will be asked to add tests before merging.
