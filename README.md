# OpenSearch Agent Server

A multi-agent orchestration server for OpenSearch Dashboards with context-aware routing and Model Context Protocol (MCP) integration.

## Overview

OpenSearch Agent Server enables intelligent agent-based interactions within OpenSearch Dashboards by:

- **Multi-Agent Orchestration** — Routes requests to specialized agents based on context
- **OpenSearch Integration** — Connects to OpenSearch via MCP for real-time data access
- **AG-UI Protocol** — Implements OpenSearch Dashboard's agent UI protocol with SSE streaming
- **Flexible LLM Support** — Works with AWS Bedrock, Ollama, or other LLM providers
- **Production Ready** — Includes authentication, rate limiting, error recovery, and observability

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
- **OpenSearch 2.x** (local or remote cluster)
- **LLM Provider** (choose one):
  - AWS Bedrock (requires AWS credentials)
  - Ollama (local installation)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mingshl/opensearch-agent-server.git
   cd opensearch-agent-server
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Configuration

Create a `.env` file with the following settings:

```bash
# OpenSearch Connection
OPENSEARCH_URL=https://localhost:9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=admin

# Authentication (set to false for local development)
AG_UI_AUTH_ENABLED=false

# CORS (allow OpenSearch Dashboards origin)
AG_UI_CORS_ORIGINS=http://localhost:5601

# LLM Provider — Option 1: AWS Bedrock
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
BEDROCK_INFERENCE_PROFILE_ARN=arn:aws:bedrock:...

# LLM Provider — Option 2: Ollama (local)
OLLAMA_MODEL=llama3

# Logging
AG_UI_LOG_FORMAT=human
AG_UI_LOG_LEVEL=INFO
```

## Quick Start

```bash
./scripts/quickstart.sh
```

This clones, builds, and starts everything in one command:

1. Clones [search-relevance](https://github.com/opensearch-project/search-relevance) and [OpenSearch Dashboards](https://github.com/opensearch-project/OpenSearch-Dashboards) (with the [dashboards-search-relevance](https://github.com/opensearch-project/dashboards-search-relevance) plugin)
2. Bootstraps OSD and starts OpenSearch via `./gradlew run`
3. Starts MCP Server (port 3001), OSD (port 5601), and Agent Server (port 8001)
4. Creates a workspace with a local data source and loads demo data
5. Runs a smoke test against all services

**Prerequisites:** Java 21+, Node.js 20+, Python 3.12+, [uv](https://astral.sh/uv), yarn, jq, curl

**Access the Chat:** Open http://localhost:5601 and click the chat icon in the header.

### Manual Setup

To run each component separately:

**Terminal 1 - OpenSearch**
```bash
# Start OpenSearch on port 9200
docker run -d -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=Admin1234!" \
  opensearchproject/opensearch:latest

# Verify
curl http://localhost:9200 -u admin:Admin1234!
```

**Terminal 2 - Agent Server**
```bash
cd opensearch-agent-server
cp .env.example .env
# Edit .env with your settings
source .venv/bin/activate
python run_server.py

# Server starts on http://localhost:8001
```

**Terminal 3 - OpenSearch Dashboards**
```bash
cd OpenSearch-Dashboards
# Ensure config/opensearch_dashboards.yml has chat.agUiUrl configured
yarn start --no-base-path

# Dashboard opens on http://localhost:5601
```

**Access the Chat**
- Open http://localhost:5601
- Click the chat icon in the top-right header
- Start asking questions about your data!

## Usage

### Start the Server

```bash
python run_server.py
```

Or using uvicorn directly:

```bash
uvicorn server.ag_ui_app:app --host 0.0.0.0 --port 8001
```

The server will start on `http://localhost:8001`

### Verify Installation

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

### Integration with OpenSearch Dashboards

1. **Start OpenSearch** (port 9200)
   ```bash
   # Using Docker
   docker run -d -p 9200:9200 -p 9600:9600 \
     -e "discovery.type=single-node" \
     -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=Admin1234!" \
     opensearchproject/opensearch:latest

   # Or use your local OpenSearch installation
   ```

2. **Start OpenSearch Agent Server** (port 8001)
   ```bash
   cd opensearch-agent-server
   source .venv/bin/activate
   python run_server.py
   ```

3. **Configure OpenSearch Dashboards**

   Edit `config/opensearch_dashboards.yml`:

   ```yaml
   # OpenSearch connection
   opensearch.hosts: ["http://localhost:9200"]
   opensearch.ssl.verificationMode: none

   # Enable new UI header (required for chat button)
   uiSettings:
     overrides:
       "home:useNewHomePage": true

   # Enable context provider (sends page context to agent)
   contextProvider:
     enabled: true

   # Enable chat with opensearch agent server
   chat:
     enabled: true
     agUiUrl: "http://localhost:8001/runs"
   ```

4. **Start OpenSearch Dashboards** (port 5601)
   ```bash
   cd OpenSearch-Dashboards
   yarn start --no-base-path
   ```

5. **Access the Chat Interface**
   - Open http://localhost:5601 in your browser
   - Look for the chat icon in the top-right header
   - Click to open the assistant panel
   - Start chatting with your data!

## Development

### Install Development Dependencies

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

### OpenSearch Connection Issues

- Verify OpenSearch is running: `curl http://localhost:9200`
- Check credentials in `.env`
- Disable SSL verification for local development

### LLM Provider Issues

- **AWS Bedrock**: Ensure AWS credentials are configured
- **Ollama**: Verify Ollama is running: `ollama list`

### Port Conflicts

If port 8001 is in use, modify the startup command:
```bash
uvicorn server.ag_ui_app:app --host 0.0.0.0 --port 8002
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

- Built with [strands-agents](https://github.com/anthropics/strands-agents) for multi-agent orchestration
- Implements [AG-UI Protocol](https://github.com/opensearch-project/ag-ui-protocol) for OpenSearch Dashboards
- Uses [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) for OpenSearch integration
