"""AG-UI Server Package.

Provides AG-UI protocol server for the multi-agent system.
"""

from importlib.metadata import PackageNotFoundError, version

from server.agent_orchestrator import AgentOrchestrator


__version__ = version("opensearch-agent-server")


__all__ = [
    "AgentOrchestrator",
]
