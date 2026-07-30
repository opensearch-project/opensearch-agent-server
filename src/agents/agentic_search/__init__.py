# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Agentic-search agent: natural-language query -> OpenSearch DSL, via ``POST /invoke``."""

from __future__ import annotations

from agents.agentic_search.agent import (
    AgenticSearchAgent,
    create_agentic_search_agent,
)

__all__ = ["AgenticSearchAgent", "create_agentic_search_agent"]
