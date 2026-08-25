# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Unit tests for configuration management."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.config import (
    ServerConfig,
    get_config,
    reset_config,
    validate_config,
    validate_config_on_startup,
)
from server.constants import (
    DEFAULT_AUTH_ENABLED,
    DEFAULT_AUTH_MODE,
    DEFAULT_AUTH_STRATEGIES,
    DEFAULT_EVENT_LIMIT,
    DEFAULT_EVENT_QUEUE_TIMEOUT,
    DEFAULT_MAX_EVENT_QUEUE_SIZE,
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MESSAGE_LIMIT,
    DEFAULT_RATE_LIMIT_ENABLED,
    DEFAULT_RATE_LIMIT_PER_HOUR,
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    DEFAULT_RETRY_INITIAL_TIMEOUT,
    DEFAULT_RETRY_MAX_TIMEOUT,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_THREAD_LIMIT,
)


@pytest.fixture(autouse=True)
def reset_config_singleton():
    reset_config()
    yield
    reset_config()


class TestServerConfigDefaults:
    def test_default_values(self, monkeypatch):

        monkeypatch.delenv("OPENSEARCH_URL", raising=False)
        monkeypatch.delenv("PHOENIX_URL", raising=False)
        monkeypatch.delenv("PHOENIX_PUBLIC_URL", raising=False)

        config = ServerConfig()

        assert config.server_host == DEFAULT_SERVER_HOST
        assert config.server_port == DEFAULT_SERVER_PORT
        assert config.max_request_body_bytes == DEFAULT_MAX_REQUEST_BODY_BYTES
        assert config.opensearch_url == "http://localhost:9200"
        assert config.phoenix_public_url is None
        assert config.cors_origins is None
        assert config.cors_methods is None
        assert config.cors_headers is None
        assert config.enable_persistence is False
        assert config.db_path == ".ag-ui/chat_history.db"
        assert config.rate_limit_enabled == DEFAULT_RATE_LIMIT_ENABLED
        assert config.rate_limit_per_minute == DEFAULT_RATE_LIMIT_PER_MINUTE
        assert config.rate_limit_per_hour == DEFAULT_RATE_LIMIT_PER_HOUR
        assert config.max_event_queue_size == DEFAULT_MAX_EVENT_QUEUE_SIZE
        assert config.default_thread_limit == DEFAULT_THREAD_LIMIT
        assert config.default_message_limit == DEFAULT_MESSAGE_LIMIT
        assert config.default_event_limit == DEFAULT_EVENT_LIMIT
        assert config.phoenix_url == "http://phoenix:6006"
        assert config.log_format == "human"
        assert config.log_level == "INFO"
        assert config.max_generator_wait_time == 1800.0
        assert config.max_consecutive_timeouts == 1800
        assert config.event_queue_timeout == DEFAULT_EVENT_QUEUE_TIMEOUT
        assert config.max_retries == DEFAULT_MAX_RETRIES
        assert config.retry_initial_timeout == DEFAULT_RETRY_INITIAL_TIMEOUT
        assert config.retry_max_timeout == DEFAULT_RETRY_MAX_TIMEOUT
        assert config.auth_enabled == DEFAULT_AUTH_ENABLED
        assert config.auth_mode == DEFAULT_AUTH_MODE
        assert config.auth_strategies == DEFAULT_AUTH_STRATEGIES
        assert config.jwt_secret is None
        assert config.jwt_public_key is None
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_user_id_claim == "sub"
        assert config.api_keys is None
        assert config.trusted_proxy_enabled is False


class TestEnvironmentVariables:
    def test_ag_ui_prefixed_env_vars(self, monkeypatch):

        monkeypatch.setenv("AG_UI_SERVER_HOST", "127.0.0.1")
        monkeypatch.setenv("AG_UI_SERVER_PORT", "9000")
        monkeypatch.setenv("AG_UI_ENABLE_PERSISTENCE", "true")
        monkeypatch.setenv("AG_UI_LOG_LEVEL", "DEBUG")

        config = ServerConfig()

        assert config.server_host == "127.0.0.1"
        assert config.server_port == 9000
        assert config.enable_persistence is True
        assert config.log_level == "DEBUG"

    def test_non_prefixed_env_vars(self, monkeypatch):
        monkeypatch.setenv("OPENSEARCH_URL", "https://test-url:6969")
        monkeypatch.setenv("PHOENIX_URL", "https://test-url:6769")
        monkeypatch.setenv("PHOENIX_PUBLIC_URL", "https://test-url:6767")

        config = ServerConfig()

        assert config.opensearch_url == "https://test-url:6969"
        assert config.phoenix_url == "https://test-url:6769"
        assert config.phoenix_public_url == "https://test-url:6767"


class TestCorsConfiguration:
    def test_cors_origins_list_unset(self):
        config = ServerConfig(cors_origins=None)
        assert config.get_cors_origins_list() == []

    def test_cors_origins_list_set(self):
        config = ServerConfig(
            cors_origins="https://test-url:6969, https://test-url:6767"
        )
        assert config.get_cors_origins_list() == [
            "https://test-url:6969",
            "https://test-url:6767",
        ]

    def test_cors_origins_list_wildcard(self):
        config = ServerConfig(cors_origins="https://test-url:6969, *")
        assert config.get_cors_origins_list() == ["*"]

    def test_get_cors_methods_list_unset(self):
        config = ServerConfig(cors_methods=None)
        assert config.get_cors_methods_list() == ["GET", "POST", "OPTIONS"]

    def test_get_cors_methods_list_set(self):
        config = ServerConfig(cors_methods="POST, PUT, OPTIONS")
        assert config.get_cors_methods_list() == ["POST", "PUT", "OPTIONS"]

    def test_get_cors_headers_list_unset(self):
        config = ServerConfig(cors_headers=None)
        assert config.get_cors_headers_list() == [
            "Content-Type",
            "Accept",
            "Authorization",
            "X-User-Id",
        ]

    def test_get_cors_headers_list_set(self):
        config = ServerConfig(cors_headers="Content-Type, Authorization")
        assert config.get_cors_headers_list() == ["Content-Type", "Authorization"]


class TestFieldValidation:
    @pytest.mark.parametrize("invalid_format", ["xml", "yaml", "csv"])
    def test_invalid_log_format_rejected(self, invalid_format):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(log_format=invalid_format)
        assert "log_format must be 'human' or 'json'" in str(exc.value)

    @pytest.mark.parametrize("invalid_level", ["VERBOSE", "TRACE", "QUIET"])
    def test_invalid_log_level_rejected(self, invalid_level):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(log_level=invalid_level)
        assert "log_level must be one of" in str(exc.value)

    @pytest.mark.parametrize("invalid_mode", ["bypass", "none", "INVALID"])
    def test_invalid_auth_mode_rejected(self, invalid_mode):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(auth_mode=invalid_mode)
        assert "auth_mode must be 'strict' or 'permissive'" in str(exc.value)

    @pytest.mark.parametrize("invalid_alg", ["MD5", "SHA256", "ES256"])
    def test_invalid_jwt_algorithm_rejected(self, invalid_alg):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(jwt_algorithm=invalid_alg)
        assert "jwt_algorithm must be 'HS256' or 'RS256'" in str(exc.value)


class TestJwtAuthStrategy:
    def test_jwt_hs356_has_jwt(self):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(
                auth_strategies="token", jwt_algorithm="HS256", jwt_secret=None
            )
        assert (
            "jwt_secret is required when using JWT authentication with HS256 algorithm"
            in str(exc.value)
        )

    def test_jwt_rs256_has_jwt(self):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(
                auth_strategies="token", jwt_algorithm="RS256", jwt_public_key=None
            )
        assert (
            "jwt_public_key is required when using JWT authentication with RS256 algorithm"
            in str(exc.value)
        )

    def test_apiKey_has_apiKey(self):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(auth_strategies="apiKey", api_keys=None)
        assert "api_keys is required when using API key authentication" in str(
            exc.value
        )

    def test_validate_auth_config(self):

        config = ServerConfig(
            auth_strategies="TOKEN, APIKEY",
            jwt_algorithm="HS256",
            jwt_secret="jwt-test-token-key-1234",
            api_keys="apikeysecretstring",
        )

        assert config.auth_strategies == "TOKEN, APIKEY"
        assert config.jwt_secret == "jwt-test-token-key-1234"
        assert config.api_keys == "apikeysecretstring"


class TestRateLimitBounds:
    @pytest.mark.parametrize("invalid_port", [0, -1, 70000])
    def test_invalid_server_port_bounds(self, invalid_port):
        with pytest.raises(ValidationError):
            ServerConfig(server_port=invalid_port)

    @pytest.mark.parametrize("invalid_rate", [0, -1, -60])
    def test_invalid_rate_limit_per_minute_bounds(self, invalid_rate):
        with pytest.raises(ValidationError):
            ServerConfig(rate_limit_per_minute=invalid_rate)

    @pytest.mark.parametrize("invalid_rate", [0, -100])
    def test_invalid_rate_limit_per_hour_bounds(self, invalid_rate):
        with pytest.raises(ValidationError):
            ServerConfig(rate_limit_per_hour=invalid_rate)


class TestStartupConfigValidation:
    def test_auth_enabled_in_production_with_no_proxy_errors(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        config = ServerConfig(
            auth_enabled=True, auth_strategies="header", trusted_proxy_enabled=False
        )

        with pytest.raises(ValueError) as exc:
            validate_config_on_startup(config)
        assert "Header authentication requires trusted proxy in production" in str(
            exc.value
        )

    def test_auth_enabled_in_development_with_no_proxy_logs(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "development")
        config = ServerConfig(
            auth_enabled=True, auth_strategies="header", trusted_proxy_enabled=False
        )

        issues = validate_config(config)

        assert any(severity == "warning" for severity, _ in issues)
        assert not any(severity == "error" for severity, _ in issues)


class TestSingletonLifecycle:
    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_clears_singleton(self, monkeypatch):
        cfg1 = get_config()
        assert cfg1.server_port == 8001

        reset_config()
        monkeypatch.setenv("AG_UI_SERVER_PORT", "9999")
        cfg2 = get_config()

        assert cfg1 is not cfg2
        assert cfg2.server_port == 9999
