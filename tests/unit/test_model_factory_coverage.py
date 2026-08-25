# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for utils.model_factory.

BedrockModel/OllamaModel constructors and boto3.Session are patched; no AWS
or Ollama connections are made.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from utils.model_factory import _DEFAULT_BEDROCK_MODEL_ID, create_model, get_provider

pytestmark = pytest.mark.unit


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


class TestGetProvider:
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
