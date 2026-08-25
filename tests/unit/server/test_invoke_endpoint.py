# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Unit tests for the /invoke non-streaming endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from server.ag_ui_app import app, get_orchestrator

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_orchestrator():
    """Create a mock AgentOrchestrator with a working invoke method."""
    orch = MagicMock()
    orch.invoke = AsyncMock(return_value="This is the agent response.")
    return orch


@pytest.fixture
def client(mock_orchestrator):
    """FastAPI TestClient with orchestrator dependency overridden."""
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    with patch("server.ag_ui_app.rate_limit", lambda f: f):
        yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


class TestInvokeEndpoint:
    """Tests for POST /invoke."""

    def test_query_string_returns_success(self, client, mock_orchestrator):
        """Simple string query returns the agent response."""
        response = client.post("/invoke", json={"query": "What indexes exist?"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["response"] == "This is the agent response."
        mock_orchestrator.invoke.assert_called_once_with(
            prompt="What indexes exist?",
            agent_name=None,
            headers=None,
            timeout=600,
            context=None,
        )

    def test_messages_list_returns_success(self, client, mock_orchestrator):
        """Message list input is converted to Strands format."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        response = client.post("/invoke", json={"messages": messages})

        assert response.status_code == 200
        assert response.json()["status"] == "success"

        expected_prompt = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi there"}]},
        ]
        mock_orchestrator.invoke.assert_called_once_with(
            prompt=expected_prompt,
            agent_name=None,
            headers=None,
            timeout=600,
            context=None,
        )

    def test_explicit_agent_name(self, client, mock_orchestrator):
        """Agent name from request body is passed to orchestrator."""
        response = client.post("/invoke", json={"query": "hi", "agent": "decomposer"})

        assert response.status_code == 200
        mock_orchestrator.invoke.assert_called_once_with(
            prompt="hi",
            agent_name="decomposer",
            headers=None,
            timeout=600,
            context=None,
        )

    def test_missing_query_and_messages_returns_400(self, client, mock_orchestrator):
        """Request without query or messages returns 400."""
        response = client.post("/invoke", json={"agent": "default"})

        assert response.status_code == 400
        body = response.json()
        assert body["status"] == "error"
        assert body["error_type"] == "ValidationError"
        assert "query" in body["error"]
        mock_orchestrator.invoke.assert_not_called()

    def test_unknown_agent_returns_500(self, client, mock_orchestrator):
        """RuntimeError from orchestrator (unknown agent) returns 500."""
        mock_orchestrator.invoke = AsyncMock(
            side_effect=RuntimeError(
                "No agent factory registered with name 'bad'. Available: ['default']"
            )
        )

        response = client.post("/invoke", json={"query": "hi", "agent": "bad"})

        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "error"
        assert body["error_type"] == "RuntimeError"
        assert "bad" in body["error"]

    def test_error_response_has_no_traceback(self, client, mock_orchestrator):
        """Error responses must not leak tracebacks."""
        mock_orchestrator.invoke = AsyncMock(side_effect=ValueError("something broke"))

        response = client.post("/invoke", json={"query": "hi"})

        body = response.json()
        assert "traceback" not in body

    def test_auth_headers_forwarded(self, client, mock_orchestrator):
        """Authorization header is forwarded to the orchestrator."""
        response = client.post(
            "/invoke",
            json={"query": "hi"},
            headers={"Authorization": "Bearer test-token-123"},
        )

        assert response.status_code == 200
        mock_orchestrator.invoke.assert_called_once_with(
            prompt="hi",
            agent_name=None,
            headers={"authorization": "Bearer test-token-123"},
            timeout=600,
            context=None,
        )

    def test_empty_agent_response(self, client, mock_orchestrator):
        """Empty agent response still returns success."""
        mock_orchestrator.invoke = AsyncMock(return_value="")

        response = client.post("/invoke", json={"query": "hi"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["response"] == ""

    def test_timeout_returns_408(self, client, mock_orchestrator):
        """Request that exceeds timeout returns 408."""
        mock_orchestrator.invoke = AsyncMock(
            side_effect=TimeoutError("Agent 'default' did not complete within 1s.")
        )

        response = client.post("/invoke", json={"query": "hi", "timeout": 1})

        assert response.status_code == 408
        body = response.json()
        assert body["status"] == "error"
        assert body["error_type"] == "TimeoutError"

    def test_custom_timeout_passed_to_orchestrator(self, client, mock_orchestrator):
        """Custom timeout value from request body is passed to orchestrator."""
        response = client.post("/invoke", json={"query": "hi", "timeout": 300})

        assert response.status_code == 200
        mock_orchestrator.invoke.assert_called_once_with(
            prompt="hi",
            agent_name=None,
            headers=None,
            timeout=300,
            context=None,
        )

    def test_default_timeout_is_600(self, client, mock_orchestrator):
        """Default timeout is 600 seconds when not specified."""
        response = client.post("/invoke", json={"query": "hi"})

        assert response.status_code == 200
        mock_orchestrator.invoke.assert_called_once_with(
            prompt="hi",
            agent_name=None,
            headers=None,
            timeout=600,
            context=None,
        )

    def test_context_forwarded_to_orchestrator(self, client, mock_orchestrator):
        """The structured `context` object is passed through to the orchestrator."""
        context = {"index_name": "products", "template_id": "product_search"}
        response = client.post(
            "/invoke",
            json={"query": "red shoes", "agent": "agentic_search", "context": context},
        )

        assert response.status_code == 200
        mock_orchestrator.invoke.assert_called_once_with(
            prompt="red shoes",
            agent_name="agentic_search",
            headers=None,
            timeout=600,
            context=context,
        )

    def test_response_format_inference_results_wraps_reply(
        self, client, mock_orchestrator
    ):
        """response_format=inference_results wraps the reply in the ml-commons envelope."""
        dsl = '{"query":{"match_all":{}}}'
        mock_orchestrator.invoke = AsyncMock(return_value=dsl)

        response = client.post(
            "/invoke",
            json={
                "query": "everything",
                "agent": "agentic_search",
                "context": {"index_name": "idx"},
                "response_format": "inference_results",
            },
        )

        assert response.status_code == 200
        body = response.json()
        # Enveloped shape, not {response, status}: the DSL is a string at
        # output[0].result so a connector's passthrough can read it.
        assert "response" not in body
        result = body["inference_results"][0]["output"][0]["result"]
        assert result == dsl
        assert isinstance(result, str)

    def test_default_response_format_unchanged(self, client, mock_orchestrator):
        """Without response_format, the default {response, status} shape is returned."""
        response = client.post(
            "/invoke", json={"query": "hi", "response_format": "text"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["response"] == "This is the agent response."
        assert "inference_results" not in body
