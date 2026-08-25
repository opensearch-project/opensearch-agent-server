# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Unit tests for the LLM model factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from utils.model_factory import _DEFAULT_BEDROCK_MODEL_ID, create_model, get_provider

pytestmark = pytest.mark.unit


class TestGetProvider:
    """Tests for provider auto-detection."""

    def test_returns_bedrock_by_default(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        assert get_provider() == "bedrock"

    def test_returns_ollama_when_ollama_model_set(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "llama3")
        assert get_provider() == "ollama"


class TestCreateModelBedrock:
    """Tests for Bedrock model creation."""

    @pytest.fixture(autouse=True)
    def _use_bedrock(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.setenv("BEDROCK_MODEL", "arn:aws:bedrock:us-east-1:123:model/test")
        monkeypatch.setenv(
            "BEDROCK_SMALL_MODEL", "arn:aws:bedrock:us-east-1:123:model/haiku"
        )

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_creates_default_bedrock_model(self, mock_session_cls, mock_init):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        create_model()

        mock_init.assert_called_once_with(
            model_id="arn:aws:bedrock:us-east-1:123:model/test",
            boto_session=mock_session,
            streaming=True,
        )

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_creates_small_bedrock_model(self, mock_session_cls, mock_init):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        create_model(tier="small")

        mock_init.assert_called_once_with(
            model_id="arn:aws:bedrock:us-east-1:123:model/haiku",
            boto_session=mock_session,
            streaming=True,
        )


class TestCreateModelBedrockFallbacks:
    """Tests for Bedrock model creation fallback chain."""

    @pytest.fixture(autouse=True)
    def _no_ollama(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("BEDROCK_MODEL", raising=False)
        monkeypatch.delenv("BEDROCK_SMALL_MODEL", raising=False)
        monkeypatch.delenv("BEDROCK_INFERENCE_PROFILE_ARN", raising=False)
        monkeypatch.delenv("BEDROCK_HAIKU_INFERENCE_PROFILE_ARN", raising=False)

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_default_tier_falls_back_to_default_model_id(
        self, mock_session_cls, mock_init
    ):
        """Default tier with no env vars uses _DEFAULT_BEDROCK_MODEL_ID."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        create_model()

        mock_init.assert_called_once_with(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            boto_session=mock_session,
            streaming=True,
        )

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_small_tier_falls_back_to_bedrock_model(
        self, mock_session_cls, mock_init, monkeypatch
    ):
        """Small tier with only BEDROCK_MODEL set falls back to it."""
        monkeypatch.setenv(
            "BEDROCK_MODEL", "arn:aws:bedrock:us-east-1:123:model/sonnet"
        )
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        create_model(tier="small")

        mock_init.assert_called_once_with(
            model_id="arn:aws:bedrock:us-east-1:123:model/sonnet",
            boto_session=mock_session,
            streaming=True,
        )

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_small_tier_falls_back_to_default_model_id(
        self, mock_session_cls, mock_init
    ):
        """Small tier with neither env var set uses _DEFAULT_BEDROCK_MODEL_ID."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        create_model(tier="small")

        mock_init.assert_called_once_with(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            boto_session=mock_session,
            streaming=True,
        )

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_default_tier_falls_back_to_legacy_env_var(
        self, mock_session_cls, mock_init, monkeypatch
    ):
        """Default tier with legacy BEDROCK_INFERENCE_PROFILE_ARN still works."""
        monkeypatch.setenv(
            "BEDROCK_INFERENCE_PROFILE_ARN",
            "arn:aws:bedrock:us-east-1:123:inference-profile/my-profile",
        )
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        create_model()

        mock_init.assert_called_once_with(
            model_id="arn:aws:bedrock:us-east-1:123:inference-profile/my-profile",
            boto_session=mock_session,
            streaming=True,
        )

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_small_tier_falls_back_to_legacy_haiku_env_var(
        self, mock_session_cls, mock_init, monkeypatch
    ):
        """Small tier with legacy BEDROCK_HAIKU_INFERENCE_PROFILE_ARN still works."""
        monkeypatch.setenv(
            "BEDROCK_HAIKU_INFERENCE_PROFILE_ARN",
            "arn:aws:bedrock:us-east-1:123:inference-profile/haiku",
        )
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        create_model(tier="small")

        mock_init.assert_called_once_with(
            model_id="arn:aws:bedrock:us-east-1:123:inference-profile/haiku",
            boto_session=mock_session,
            streaming=True,
        )


class TestCreateModelOllama:
    """Tests for Ollama model creation."""

    @pytest.fixture(autouse=True)
    def _use_ollama(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "llama3")
        monkeypatch.delenv("OLLAMA_SMALL_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)

    @patch("strands.models.ollama.OllamaModel.__init__", return_value=None)
    def test_creates_default_ollama_model(self, mock_init):
        create_model()

        mock_init.assert_called_once_with(
            host="http://localhost:11434",
            model_id="llama3",
        )

    @patch("strands.models.ollama.OllamaModel.__init__", return_value=None)
    def test_creates_small_ollama_model_fallback(self, mock_init):
        """Small tier falls back to OLLAMA_MODEL when OLLAMA_SMALL_MODEL is not set."""
        create_model(tier="small")

        mock_init.assert_called_once_with(
            host="http://localhost:11434",
            model_id="llama3",
        )

    @patch("strands.models.ollama.OllamaModel.__init__", return_value=None)
    def test_creates_small_ollama_model_explicit(self, mock_init, monkeypatch):
        monkeypatch.setenv("OLLAMA_SMALL_MODEL", "phi3")

        create_model(tier="small")

        mock_init.assert_called_once_with(
            host="http://localhost:11434",
            model_id="phi3",
        )

    @patch("strands.models.ollama.OllamaModel.__init__", return_value=None)
    def test_uses_custom_host(self, mock_init, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://remote:11434")

        create_model()

        mock_init.assert_called_once_with(
            host="http://remote:11434",
            model_id="llama3",
        )


# --- additional coverage tests (merged) ---
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for utils.model_factory.

BedrockModel/OllamaModel constructors and boto3.Session are patched; no AWS
or Ollama connections are made.
"""


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "OLLAMA_MODEL",
        "OLLAMA_SMALL_MODEL",
        "OLLAMA_HOST",
        "BEDROCK_MODEL",
        "BEDROCK_SMALL_MODEL",
        "BEDROCK_INFERENCE_PROFILE_ARN",
        "BEDROCK_HAIKU_INFERENCE_PROFILE_ARN",
        "LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


class TestGetProviderResolution:
    def test_bedrock_by_default(self, clean_env):
        assert get_provider() == "bedrock"

    def test_ollama_when_set(self, clean_env):
        clean_env.setenv("OLLAMA_MODEL", "llama3")
        assert get_provider() == "ollama"


class TestBedrock:
    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_default_falls_back_to_default_model_id(
        self, mock_session, mock_init, clean_env
    ):
        create_model()
        assert mock_init.call_args[1]["model_id"] == _DEFAULT_BEDROCK_MODEL_ID
        assert mock_init.call_args[1]["streaming"] is True

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_explicit_bedrock_model(self, mock_session, mock_init, clean_env):
        clean_env.setenv("BEDROCK_MODEL", "arn:model:main")
        create_model()
        assert mock_init.call_args[1]["model_id"] == "arn:model:main"

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_small_tier_prefers_small_model(self, mock_session, mock_init, clean_env):
        clean_env.setenv("BEDROCK_SMALL_MODEL", "arn:model:haiku")
        create_model(tier="small")
        assert mock_init.call_args[1]["model_id"] == "arn:model:haiku"

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_temperature_applied_when_valid(self, mock_session, mock_init, clean_env):
        clean_env.setenv("LLM_TEMPERATURE", "0.7")
        create_model()
        assert mock_init.call_args[1]["temperature"] == 0.7

    @patch("strands.models.bedrock.BedrockModel.__init__", return_value=None)
    @patch("boto3.Session")
    def test_invalid_temperature_ignored(self, mock_session, mock_init, clean_env):
        clean_env.setenv("LLM_TEMPERATURE", "not-a-float")
        create_model()
        assert "temperature" not in mock_init.call_args[1]


class TestOllama:
    @patch("strands.models.ollama.OllamaModel.__init__", return_value=None)
    def test_ollama_default(self, mock_init, clean_env):
        clean_env.setenv("OLLAMA_MODEL", "llama3")
        create_model()
        assert mock_init.call_args[1]["model_id"] == "llama3"
        assert mock_init.call_args[1]["host"] == "http://localhost:11434"

    @patch("strands.models.ollama.OllamaModel.__init__", return_value=None)
    def test_ollama_small_falls_back_to_main(self, mock_init, clean_env):
        clean_env.setenv("OLLAMA_MODEL", "llama3")
        create_model(tier="small")
        assert mock_init.call_args[1]["model_id"] == "llama3"

    @patch("strands.models.ollama.OllamaModel.__init__", return_value=None)
    def test_ollama_small_uses_small_model(self, mock_init, clean_env):
        clean_env.setenv("OLLAMA_MODEL", "llama3")
        clean_env.setenv("OLLAMA_SMALL_MODEL", "llama3-small")
        clean_env.setenv("OLLAMA_HOST", "http://ollama:9999")
        create_model(tier="small")
        assert mock_init.call_args[1]["model_id"] == "llama3-small"
        assert mock_init.call_args[1]["host"] == "http://ollama:9999"
