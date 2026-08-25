# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.logging_config.

Covers the JSON formatter, request-id filter/contextvar, logging configuration
(force / no-force / json), and the env + config accessors. The root logger's
handler state is saved and restored so this file does not leak configuration
into other tests.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from server.logging_config import (
    JSONFormatter,
    RequestIdFilter,
    configure_logging,
    get_logging_config,
    get_logging_config_from_env,
    request_id_contextvar,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def restore_root_logger():
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def _make_record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="server.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


class TestJSONFormatter:
    def test_basic_fields(self):
        out = JSONFormatter().format(_make_record())
        assert '"message": "hello world"' in out
        assert '"level": "INFO"' in out
        assert '"logger": "server.test"' in out

    def test_includes_extra_serializable_field(self):
        out = JSONFormatter().format(_make_record(run_id="r1"))
        assert '"run_id": "r1"' in out

    def test_non_serializable_field_stringified(self):
        class Weird:
            def __str__(self) -> str:
                return "weird-obj"

        out = JSONFormatter().format(_make_record(obj=Weird()))
        assert "weird-obj" in out

    def test_underscore_prefixed_field_excluded(self):
        out = JSONFormatter().format(_make_record(_private="secret"))
        assert "secret" not in out

    def test_exception_info_included(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = _make_record()
            record.exc_info = sys.exc_info()
            out = JSONFormatter().format(record)
        assert "exception" in out
        assert "boom" in out


class TestRequestIdFilter:
    def test_injects_request_id_when_set(self):
        token = request_id_contextvar.set("req-123")
        try:
            record = _make_record()
            assert RequestIdFilter().filter(record) is True
            assert record.request_id == "req-123"
        finally:
            request_id_contextvar.reset(token)

    def test_uses_dash_when_unset(self):
        request_id_contextvar.set(None)
        record = _make_record()
        assert RequestIdFilter().filter(record) is True
        assert record.request_id == "-"


class TestConfigureLogging:
    def test_configure_json_forced(self, restore_root_logger):
        configure_logging(use_json=True, log_level="DEBUG", force=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert any(
            isinstance(h.formatter, JSONFormatter) for h in root.handlers if h.formatter
        )

    def test_configure_human_forced(self, restore_root_logger):
        configure_logging(use_json=False, log_level="WARNING", force=True)
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert root.handlers

    def test_no_force_keeps_existing_handlers(self, restore_root_logger):
        root = logging.getLogger()
        configure_logging(use_json=False, log_level="INFO", force=True)
        handler_count = len(root.handlers)
        # force=False with handlers present should not clear/add.
        configure_logging(use_json=True, log_level="INFO", force=False)
        assert len(root.handlers) == handler_count

    def test_invalid_level_defaults_to_info(self, restore_root_logger):
        configure_logging(use_json=False, log_level="NOTALEVEL", force=True)
        assert logging.getLogger().level == logging.INFO


class TestGetLoggingConfigFromEnv:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("AG_UI_LOG_FORMAT", raising=False)
        monkeypatch.delenv("AG_UI_LOG_LEVEL", raising=False)
        use_json, level = get_logging_config_from_env()
        assert use_json is False
        assert level == "INFO"

    def test_json_and_level(self, monkeypatch):
        monkeypatch.setenv("AG_UI_LOG_FORMAT", "JSON")
        monkeypatch.setenv("AG_UI_LOG_LEVEL", "debug")
        use_json, level = get_logging_config_from_env()
        assert use_json is True
        assert level == "DEBUG"


class TestGetLoggingConfig:
    def test_reads_from_config(self):
        fake_cfg = SimpleNamespace(log_format="json", log_level="ERROR")
        with patch("server.logging_config.get_config", return_value=fake_cfg):
            use_json, level = get_logging_config()
        assert use_json is True
        assert level == "ERROR"

    def test_human_format(self):
        fake_cfg = SimpleNamespace(log_format="human", log_level="INFO")
        with patch("server.logging_config.get_config", return_value=fake_cfg):
            use_json, level = get_logging_config()
        assert use_json is False
        assert level == "INFO"
