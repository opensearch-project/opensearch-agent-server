# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""AG-UI Server Package.

Provides AG-UI protocol server for the multi-agent system.
"""

from importlib.metadata import version

from server.agent_orchestrator import AgentOrchestrator

__version__ = version("opensearch-agent-server")


__all__ = [
    "AgentOrchestrator",
]
