"""
Pytest configuration and shared fixtures for opensearch-agent-server tests.
"""

import os
from collections.abc import Callable

import pytest

# ---------------------------------------------------------------------------
# Environment variables — set before any imports that read config at module level
# ---------------------------------------------------------------------------
os.environ["OPENSEARCH_URL"] = os.getenv("OPENSEARCH_URL", "http://localhost:9200")

# Mock AWS credentials for tests (real calls are patched out in unit tests)
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "test-key")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "test-secret")
os.environ["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_env(monkeypatch: pytest.MonkeyPatch) -> Callable[..., dict[str, str]]:
    """Patch environment variables for the duration of a test.

    Usage:
        def test_something(patch_env):
            patch_env(OPENSEARCH_URL="http://other:9200")
    """

    def _patch(clear: bool = False, **kwargs: str) -> dict[str, str]:
        if clear:
            for key in dict(os.environ):
                monkeypatch.delenv(key, raising=False)
        for key, value in kwargs.items():
            monkeypatch.setenv(key, value)
        return kwargs

    return _patch


@pytest.fixture
def test_opensearch_url() -> str:
    """Returns the test OpenSearch URL (TEST_OPENSEARCH_URL env var, default localhost:9200)."""
    return os.getenv("TEST_OPENSEARCH_URL", "http://localhost:9200")
