# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Base types for sub-agents.

Defines the protocol that sub-agent factories should follow.
"""

from __future__ import annotations

from typing import Protocol

from strands import Agent


class SubAgentFactory(Protocol):
    """Protocol for sub-agent factories."""

    async def create(self, opensearch_url: str) -> Agent:
        """Create the sub-agent.

        Args:
            opensearch_url: OpenSearch cluster URL.

        Returns:
            A configured Strands Agent.
        """
        ...

    @property
    def name(self) -> str:
        """Agent name."""
        ...

    @property
    def page_contexts(self) -> list[str]:
        """Page contexts this agent handles."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of this agent's capabilities."""
        ...
