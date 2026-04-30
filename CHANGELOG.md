# CHANGELOG

## [Unreleased]
### Added
- PPL reference skill and skills auto-discovery for the default agent
### Fixed
- Default agent now respects `BEDROCK_INFERENCE_PROFILE_ARN` env var instead of silently falling back to a hardcoded Sonnet 4 model (fixes #94)
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