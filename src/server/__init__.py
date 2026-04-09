"""AG-UI Server Package.

Provides AG-UI protocol server for the multi-agent system.
"""

from importlib.metadata import PackageNotFoundError, version

from server.agent_orchestrator import AgentOrchestrator

try:
    __version__ = version("opensearch-agent-server")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "AgentOrchestrator",
]
