# OpenSearch Agent Server

A multi-agent orchestration server for OpenSearch Dashboards with context-aware routing and Model Context Protocol (MCP) integration.

## Overview

OpenSearch Agent Server enables intelligent agent-based interactions within OpenSearch Dashboards by:

- **Multi-Agent Orchestration** — Routes requests to specialized agents based on context
- **OpenSearch Integration** — Connects to OpenSearch via MCP for real-time data access
- **AG-UI Protocol** — Implements OpenSearch Dashboard's agent UI protocol with SSE streaming
- **LLM Support** — AWS Bedrock today; Ollama support in progress
- **Production Ready** — Includes authentication, rate limiting, error recovery, and observability

## Demo
https://github.com/user-attachments/assets/d465d805-40c9-4158-8e4b-0805c675df45

## Architecture

```
OpenSearch Dashboards (AG-UI)
            ↓
    OpenSearch Agent Server
    ├── Router (context-based)
    ├── Agent Registry
    │   ├── ART Agent (strands-agents)
    │   └── Default Agent
    └── OpenSearch MCP Server
            ↓
    OpenSearch Cluster
```

## Features

- **Context-Aware Routing** — Automatically selects the appropriate agent based on request context
- **Streaming Responses** — Real-time SSE streaming for interactive user experiences
- **Tool Execution** — Agents can execute tools and visualize results in the dashboard
- **Authentication & Authorization** — JWT-based auth with configurable policies
- **Rate Limiting** — Protects backend services from overload
- **Error Recovery** — Automatic retry with exponential backoff
- **Observability** — Structured logging with request tracking

## Prerequisites

- **Python 3.12+**
- **OpenSearch 2.x or 3.x** (local or remote cluster)
- **AWS Bedrock** with access to a Claude inference profile.
- **For Dashboards integration**: OpenSearch Dashboards **≥ 3.6**, matched to your OpenSearch major version. Matching exact versions avoids edge cases.

## Get Started

Pick the path that matches your situation:

| You want to... | Use this section |
|---|---|
| Run the agent server against an existing OpenSearch cluster | [Option A — Install from PyPI](#option-a--install-from-pypi) |
| See everything (OS + OSD + agent + demo data) running end-to-end | [Option B — Full demo quickstart](#option-b--full-demo-quickstart) |
| Hack on this repository's source code | [Option C — Run from source](#option-c--run-from-source) |

### Option A — Install from PyPI

If you already have an OpenSearch cluster running and don't need the full quickstart setup, install the agent server directly from PyPI:

```bash
pip install opensearch-agent-server
```

> **Note:** the `opensearch-agent-server` CLI entry point is available in **v0.2.1 and later**. If `pip install` currently gives you v0.2.0, the command below will not be installed. Until v0.2.1 is published to PyPI, install from the `main` branch instead:
>
> ```bash
> pip install git+https://github.com/opensearch-project/opensearch-agent-server.git
> ```
>
> Verify with `pip show opensearch-agent-server | grep Version`.

Configure your environment:

```bash
export OPENSEARCH_URL=https://localhost:9200
export OPENSEARCH_USERNAME=admin
export OPENSEARCH_PASSWORD=admin
export AG_UI_AUTH_ENABLED=false
export AWS_REGION=us-east-1
export BEDROCK_INFERENCE_PROFILE_ARN=arn:aws:bedrock:...
```

Start the agent server and MCP server together:

```bash
opensearch-agent-server --with-mcp
```

This starts both the OpenSearch MCP Server (port 3001) and the Agent Server (port 8001) in a single process. Both stop together on `Ctrl+C`.

```bash
# Verify
curl http://localhost:8001/health    # {"status": "ok"}
curl http://localhost:8001/agents    # list registered agents
```

You can also customize the MCP server port and config:

```bash
opensearch-agent-server --with-mcp --mcp-port 3002 --mcp-config ./custom_mcp.yml
```

### Option B — Full demo quickstart

```bash
./scripts/quickstart.sh
```

This clones, builds, and starts everything in one command:

1. Clones [search-relevance](https://github.com/opensearch-project/search-relevance) and [OpenSearch Dashboards](https://github.com/opensearch-project/OpenSearch-Dashboards) (with the [dashboards-search-relevance](https://github.com/opensearch-project/dashboards-search-relevance) plugin)
2. Bootstraps OSD and starts OpenSearch via `./gradlew run`
3. Starts MCP Server (port 3001), OSD (port 5601), and Agent Server (port 8001)
4. Creates a workspace with a local data source and loads demo data
5. Runs a smoke test against all services

**Additional prerequisites for this path:** Java 21+, Node.js 20+, [uv](https://astral.sh/uv), yarn, jq, curl.

**Access the Chat:** Open http://localhost:5601 and click the chat icon in the header.

### Option C — Run from source

Use this path if you're contributing to this repository or want to modify the agent server itself. You'll need a running OpenSearch cluster and MCP server — see [Option B](#option-b--full-demo-quickstart) for a one-command setup.

1. **Clone the repository**
   ```bash
   git clone https://github.com/opensearch-project/opensearch-agent-server.git
   cd opensearch-agent-server
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in editable mode**
   ```bash
   pip install -e .
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start the server**
   ```bash
   python run_server.py
   ```

## Integration with OpenSearch Dashboards

The agent server works with any install path above. Configure OSD once (version **≥ 3.6** required), then point it at the running agent server.

### Authentication flow

```
Browser (logged-in user)
  │
  │ 1. Session (basic / SAML / OIDC)
  ▼
OpenSearch Dashboards (>= 3.6)
  │
  │ 2. POST /api/chat/proxy
  │    ┌─────────────────────────────────────────────┐
  │    │ chat.forwardCredentials: true               │
  │    │   → mint OBO token via /_plugins/_security  │
  │    │     /api/obo/token                          │
  │    │   → add "Authorization: Bearer <token>"     │
  │    │   (if OBO is unavailable at runtime, OSD    │
  │    │    logs "OBO token generation unavailable"  │
  │    │    and falls through with no header)        │
  │    │ chat.forwardCredentials: false (default)    │
  │    │   → no Authorization header                 │
  │    └─────────────────────────────────────────────┘
  ▼
Agent Server /runs
  │
  │ 3. Stores incoming Bearer token on OboAuth
  │    (thread-safe, per-request).
  │
  │ 4. Agent calls MCP tool
  │    httpx replays "Authorization: Bearer <token>"
  │    on every outgoing request (OboAuth.async_auth_flow).
  │    If no token was received, no Authorization header
  │    is added.
  ▼
MCP Server (OPENSEARCH_HEADER_AUTH=true)
  │
  │ 5. If Authorization Bearer is present: use it.
  │    Otherwise: fall back to OPENSEARCH_USERNAME /
  │    OPENSEARCH_PASSWORD env vars (MCP's own config).
  ▼
OpenSearch Cluster
  │
  └─ Identity enforcement:
     • OBO token → logged-in Dashboards user
       (row-level / field-level security applied)
     • No token → MCP service account
       (all users share one identity)
```

Edit `config/opensearch_dashboards.yml`:

```yaml
# OpenSearch connection
opensearch.hosts: ["http://localhost:9200"]
opensearch.ssl.verificationMode: none

# Required for the chat button to appear in the top-right header.
uiSettings:
  overrides:
    "home:useNewHomePage": true

# Sends current page context (app ID, filters, queries) to the agent.
contextProvider:
  enabled: true

# Chat -> agent server wiring.
chat:
  enabled: true
  agUiUrl: "http://localhost:8001/runs"
  # Forward the logged-in user's identity to the agent server via an
  # On-Behalf-Of token. Requires the OpenSearch security plugin with
  # OBO enabled (/_plugins/_security/api/obo/token). Leave false if
  # security is disabled; see "Credential forwarding" below.
  # forwardCredentials: true

# Workspaces (required for the agent's per-workspace data source routing).
# NOTE: workspaces are incompatible with the security plugin's multi-tenancy
# feature. If your cluster has opensearch_security.multitenancy.enabled=true,
# set it to false in opensearch.yml before enabling workspaces.
workspace.enabled: true
data_source.enabled: true
data_source.hideLocalCluster: false
```

**Start OpenSearch Dashboards:**

```bash
cd OpenSearch-Dashboards
yarn start --no-base-path
```

**Access the chat interface:**

- Open http://localhost:5601
- Click the chat icon in the top-right header
- Start chatting with your data

## Configuration

The full list of configuration keys is in `.env.example`. The most common ones:

```bash
# OpenSearch Connection
OPENSEARCH_URL=https://localhost:9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=admin

# Authentication (set to false for local development)
AG_UI_AUTH_ENABLED=false

# CORS (allow OpenSearch Dashboards origin)
AG_UI_CORS_ORIGINS=http://localhost:5601

# LLM Provider — AWS Bedrock (currently required)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
BEDROCK_INFERENCE_PROFILE_ARN=arn:aws:bedrock:...

# Ollama support is in progress and not yet wired up.
# OLLAMA_MODEL=llama3

# Logging
AG_UI_LOG_FORMAT=human
AG_UI_LOG_LEVEL=INFO
```

> `.env.example` is the authoritative source of configuration keys. The snippet above shows the common ones; see `.env.example` in the repo for the full list.

## Usage

### Verify the server is running

```bash
# Check server health
curl http://localhost:8001/health

# List available agents
curl http://localhost:8001/agents

# Test agent interaction (requires OpenSearch running)
curl -X POST http://localhost:8001/runs \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Show me recent logs",
    "context": [{"appId": "discover"}]
  }'
```

### Start options

```bash
# Via the PyPI CLI (Option A)
opensearch-agent-server --with-mcp

# Via the source entry point (Option C)
python run_server.py

# Or with uvicorn directly
uvicorn server.ag_ui_app:app --host 0.0.0.0 --port 8001
```

The server listens on `http://localhost:8001`.

## API Endpoints

### Health Check
```
GET /health
```
Returns server health status.

### List Agents
```
GET /agents
```
Returns available agents and their capabilities.

### Create Run (AG-UI Protocol)
```
POST /runs
```
Creates a new agent run with streaming responses via SSE.

### Get Run Status
```
GET /runs/{run_id}
```
Returns the status of a specific run.

## Troubleshooting

### `command not found: opensearch-agent-server` after `pip install`

The CLI entry point was added in **v0.2.1**. The current PyPI release is v0.2.0, which does not install this command. Until v0.2.1 is published, install from `main` instead:

```bash
pip install git+https://github.com/opensearch-project/opensearch-agent-server.git
```

### Chat icon doesn't appear in the OSD header

The chat button only renders when the new UI header is enabled:

```yaml
uiSettings:
  overrides:
    "home:useNewHomePage": true
```

Also confirm you're on OSD **≥ 3.6**; earlier versions ship an experimental chat plugin that doesn't integrate with this agent server.

### Workspaces don't work

Workspaces are incompatible with the security plugin's multi-tenancy feature. In your OpenSearch `opensearch.yml`:

```yaml
opensearch_security.multitenancy.enabled: false
```

Then restart OpenSearch. (Documented in OSD's own [explore plugin README](https://github.com/opensearch-project/OpenSearch-Dashboards/blob/main/src/plugins/explore/README.md).)

### Agent says UBI data is unavailable

The ART agent's user-behavior-analysis sub-agent requires `ubi_queries` and `ubi_events` indices from the [UBI plugin](https://docs.opensearch.org/latest/search-plugins/ubi/). Without them, the agent reports that UBI data is unavailable and falls back to LLM-generated judgment lists for evaluation — other ART features (hypothesis generation, pairwise experiments, search configurations) still work.

**Symptom to watch for:** the LLM may present an empty MCP tool result as a `502 Bad Gateway` message even though the underlying MCP call returned 200. If you don't need UBI features, use the **default agent** instead of the ART agent.

### OSD crashes with `Socket timeout` during long Bedrock calls

OSD's HTTP keepalive can terminate the connection while a sub-agent is still running against Bedrock, producing:

```
Error: Socket timeout
    at TLSSocket.onTimeout (.../agentkeepalive/lib/agent.js:350:23)
```

Raise the OSD-side timeouts in `opensearch_dashboards.yml`:

```yaml
opensearch.requestTimeout: 300000    # 5 min
opensearch.pingTimeout: 30000        # 30 s
```

Also consider running OSD under a process supervisor so it recovers automatically if it does crash.

### OpenSearch Connection Issues

- Verify OpenSearch is running: `curl http://localhost:9200`
- Check credentials in `.env`
- Disable SSL verification for local development. Note: data source SSL is a *separate* setting from cluster SSL; disabling `opensearch.ssl.verificationMode` does not automatically apply to data sources added via the workspace data-source UI.

### LLM Provider Issues

- **AWS Bedrock**: Ensure `AWS_REGION` and `BEDROCK_INFERENCE_PROFILE_ARN` are set, or that `AWS_PROFILE`/`AWS_ACCESS_KEY_ID` credentials are valid. The agent server does not start a fallback LLM if Bedrock is unreachable; you'll see errors on the first agent request.

### Port Conflicts

If port 8001 is in use, modify the startup command:
```bash
uvicorn server.ag_ui_app:app --host 0.0.0.0 --port 8002
```

## Development

### Install Development Dependencies

Assumes you've already completed [Option C — Run from source](#option-c--run-from-source). Add the test/lint extras:

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
ruff format .
ruff check .
```

### Project Structure

```
opensearch-agent-server/
├── src/
│   ├── agents/                    # Agent implementations
│   │   ├── art/                   # ART (Search Relevance Testing) agent
│   │   │   ├── art_agent.py       # ART orchestrator agent
│   │   │   └── specialized_agents.py  # Hypothesis, evaluation, UBI sub-agents
│   │   ├── base.py                # Agent protocol / base types
│   │   └── default_agent.py       # General OpenSearch assistant
│   ├── orchestrator/              # Routing and registry
│   │   ├── router.py              # Context-based routing
│   │   └── registry.py            # Agent registry
│   ├── server/                    # FastAPI application
│   │   ├── ag_ui_app.py           # Main FastAPI app and lifespan
│   │   ├── cli.py                 # CLI entry point (opensearch-agent-server command)
│   │   ├── agent_orchestrator.py  # Orchestrator: routes requests to agents
│   │   ├── run_routes.py          # AG-UI protocol endpoints
│   │   ├── config.py              # Configuration management
│   │   └── ...                    # Middleware, auth, rate limiting, etc.
│   ├── tools/                     # Agent tools (local computation)
│   │   └── art/                   # ART-specific tools
│   │       └── experiment_tools.py  # Experiment results aggregation
│   └── utils/                     # Shared utilities
│       ├── mcp_connection.py      # OpenSearch MCP client
│       ├── logging_helpers.py     # Structured logging
│       ├── monitored_tool.py      # Tool instrumentation wrapper
│       └── ...                    # Persistence, activity monitor, etc.
├── tests/
│   ├── helpers/                   # Shared test helpers
│   ├── integration/               # Integration tests
│   └── unit/                      # Unit tests
├── run_server.py                  # Entry point
├── pyproject.toml                 # Project metadata and dependencies
└── .env.example                   # Environment template
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Built with [strands-agents](https://github.com/strands-agents/sdk-python) for multi-agent orchestration
- Implements [AG-UI Protocol](https://github.com/opensearch-project/ag-ui-protocol) for OpenSearch Dashboards
- Uses [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) for OpenSearch integration
