# OpenSearch Agent Server Test Scripts

This directory contains shell scripts for testing the OpenSearch Agent Server endpoints.

## Prerequisites

- `curl` installed
- OpenSearch Agent Server running (default: `localhost:8001`)

## Available Scripts

### 1. Health Check
Tests the server's health status.
```bash
./scripts/test_health.sh [host:port]
```

### 2. List Agents
Lists all registered agents and their supported page contexts.
```bash
./scripts/test_agents.sh [host:port]
```

### 3. Test Agent Run (AG-UI Protocol)
Starts a conversation with an agent and streams the response.
```bash
./scripts/test_run.sh [question] [host:port] [page_context]
```

**Examples:**
```bash
# Ask the default question
./scripts/test_run.sh

# Ask about ART agent specifically
./scripts/test_run.sh "How do I use the hypothesis agent?" localhost:8001 search-relevance

# Test the default agent
./scripts/test_run.sh "Who are you?" localhost:8001 some-other-context
```

## Troubleshooting

- **Connection refused**: Ensure the server is running on the specified port.
- **422 Unprocessable Entity**: The request body format might be incorrect. The scripts use the camelCase format expected by the server.
- **401 Unauthorized**: If authentication is enabled, you'll need to modify the scripts to include appropriate headers (e.g., `Authorization: Bearer <token>` or `X-User-Id: <user-id>`).