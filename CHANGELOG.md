# CHANGELOG

## [Unreleased]
### Added
- Support using Ollama as model provider
- PPL reference skill and skills auto-discovery for the default agent
- Context management across all agents: proactive conversation summarization and offloading of large tool results, plus an API-boundary cap on client-returned tool results (#138)
- Load additional skills from `~/.config/opensearch-agent-server/skills/` (default) or any paths set in the `AG_UI_SKILL_PATHS` env var (PATH-style: `:`-separated on POSIX, `;`-separated on Windows)
- Support non-streaming `/invoke` endpoint
- `agentic_search` agent (natural-language query -> OpenSearch DSL) reachable via `POST /invoke`, with a per-request generation strategy (`context.strategy`, default `direct_dsl`) and example ml-commons connector/FLOW-agent wiring
- `/invoke` `context` (structured input forwarded to the agent) and `response_format` (opt-in ml-commons `inference_results` envelope) request fields
- `template_fill` generation strategy for `agentic_search`: fills a registered search template's Mustache parameters and renders them via `_render/template`, selected by passing `context.template_id`
- `multi_template_fill` generation strategy for `agentic_search`: pass several candidate templates as `context.template_ids` and one forced tool call both chooses a template and fills its parameters. Candidates are filtered to the request's index and capped; a request naming only one template still uses `template_fill`

### Changed
- Drop the `reason` field from the `agentic_search` free-DSL output. It was decoded on every request and only logged, never used to build the query; removing it cuts free-DSL generation latency substantially with no accuracy change

### Fixed
- Bundled skills (`ppl-reference`) are now included in the pip wheel — previously only available from git clones
- Default agent now respects `BEDROCK_INFERENCE_PROFILE_ARN` env var instead of silently falling back to a hardcoded Sonnet 4 model (fixes #94)
- Tool calls now appear in the correct position in the UI instead of at the end of the preceding text message; upgraded `ag-ui-strands` to 0.1.9 which properly emits `TEXT_MESSAGE_END` before `TOOL_CALL_START` (fixes #75)
- Fix credential cross-contamination under concurrent requests
- Bump mcp from 1.26.0 to 1.28.1 to address CVE-2026-59950, CVE-2026-52870, and CVE-2026-52869 (#95)
### Removed

## [0.2.0] - 2026-04-10
### Added
- CI workflow for testing on Linux, macOS, and Windows
- Publish release workflow with manual approval gate and PyPI publishing
- Changelog enforcer workflow for pull requests
- Add untriaged label workflow for new issues
- NOTICE.txt for Apache 2.0 compliance
- PyPI classifiers and project URLs in pyproject.toml
- Runtime `__version__` via importlib.metadata
- Initial experimental release of opensearch-agent-server
- Multi-agent orchestrator with page-context routing via AG-UI protocol
- Specialized agents for OpenSearch Dashboards pages (Discover, Dashboard, etc.)
- FastAPI-based AG-UI server with SSE streaming
- Authentication middleware with JWT support
- Rate limiting and error recovery
- MCP server integration for OpenSearch tool access
- Configurable via environment variables and YAML config
- Quickstart script for local development

### Fixed
- Hatch build config to support `python -m build` (wheel from sdist)
- Unit tests for specialized agents to mock `_mcp_tools` alongside `_mcp_client`

## [0.1.0] - 2026-03-02
### Added