# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for utils.logging_helpers structured-logging helpers."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from utils.logging_helpers import (
    STANDARD_KEYS,
    get_logger,
    log_critical_event,
    log_debug_event,
    log_error_event,
    log_event,
    log_info_event,
    log_warning_event,
)

pytestmark = pytest.mark.unit


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("my.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "my.module"

    def test_standard_keys_present(self):
        assert "run_id" in STANDARD_KEYS


class TestLogEvent:
    def test_log_event_passes_extra(self):
        logger = MagicMock(spec=logging.Logger)
        log_event(logger, logging.INFO, "msg", "ag_ui.evt", run_id="r1")
        logger.log.assert_called_once()
        _, kwargs = logger.log.call_args
        assert kwargs["extra"]["event"] == "ag_ui.evt"
        assert kwargs["extra"]["run_id"] == "r1"

    def test_log_debug_event(self):
        logger = MagicMock(spec=logging.Logger)
        log_debug_event(logger, "msg", "ag_ui.debug")
        assert logger.log.call_args[0][0] == logging.DEBUG

    def test_log_info_event(self):
        logger = MagicMock(spec=logging.Logger)
        log_info_event(logger, "msg", "ag_ui.info")
        assert logger.log.call_args[0][0] == logging.INFO


class TestLogWarningEvent:
    def test_warning_default(self):
        logger = MagicMock(spec=logging.Logger)
        log_warning_event(logger, "warn", "ag_ui.warn", count=3)
        logger.warning.assert_called_once()
        _, kwargs = logger.warning.call_args
        assert kwargs["exc_info"] is False
        assert kwargs["extra"]["count"] == 3

    def test_warning_with_exc_info(self):
        logger = MagicMock(spec=logging.Logger)
        log_warning_event(logger, "warn", "ag_ui.warn", exc_info=True)
        assert logger.warning.call_args[1]["exc_info"] is True


class TestLogErrorEvent:
    def test_error_with_string_error(self):
        logger = MagicMock(spec=logging.Logger)
        log_error_event(logger, "err", "ag_ui.err", error="oops")
        logger.error.assert_called_once()
        assert logger.error.call_args[1]["extra"]["error"] == "oops"

    def test_error_with_exception_object(self):
        logger = MagicMock(spec=logging.Logger)
        log_error_event(logger, "err", "ag_ui.err", error=ValueError("bad"))
        assert logger.error.call_args[1]["extra"]["error"] == "bad"

    def test_error_with_tuple_exc_info(self):
        logger = MagicMock(spec=logging.Logger)
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys

            exc = sys.exc_info()
            log_error_event(logger, "err", "ag_ui.err", exc_info=exc)
        # Tuple exc_info is stored into extra as well as passed through.
        assert logger.error.call_args[1]["extra"]["exc_info"] == exc

    def test_error_without_error_arg(self):
        logger = MagicMock(spec=logging.Logger)
        log_error_event(logger, "err", "ag_ui.err", error=None)
        assert "error" not in logger.error.call_args[1]["extra"]


class TestLogCriticalEvent:
    def test_critical_with_error(self):
        logger = MagicMock(spec=logging.Logger)
        log_critical_event(logger, "crit", "ag_ui.crit", error="fatal")
        logger.critical.assert_called_once()
        assert logger.critical.call_args[1]["extra"]["error"] == "fatal"

    def test_critical_without_error(self):
        logger = MagicMock(spec=logging.Logger)
        log_critical_event(logger, "crit", "ag_ui.crit", node_id="n1")
        assert logger.critical.call_args[1]["extra"]["node_id"] == "n1"
        assert "error" not in logger.critical.call_args[1]["extra"]
