# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for ``agents.base.SubAgentFactory``.

``SubAgentFactory`` is a ``typing.Protocol`` whose method/property bodies are
just ``...``.  A minimal concrete subclass delegates to ``super()`` so the
protocol's own bodies execute and are counted as covered.
"""

from __future__ import annotations

import pytest

from agents.base import SubAgentFactory

pytestmark = pytest.mark.unit


class _MinimalFactory(SubAgentFactory):
    """Concrete subclass that invokes each protocol body via ``super()``."""

    async def create(self, opensearch_url: str):
        return await super().create(opensearch_url)

    @property
    def name(self) -> str:
        return super().name

    @property
    def page_contexts(self) -> list[str]:
        return super().page_contexts

    @property
    def description(self) -> str:
        return super().description


async def test_create_body_executes() -> None:
    """The protocol's ``create`` body runs and yields ``None``."""
    factory = _MinimalFactory()
    assert await factory.create("http://localhost:9200") is None


def test_property_bodies_execute() -> None:
    """Each protocol property getter body runs and yields ``None``."""
    factory = _MinimalFactory()
    assert factory.name is None
    assert factory.page_contexts is None
    assert factory.description is None
