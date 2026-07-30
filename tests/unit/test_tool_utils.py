# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""
Unit tests for tool_utils module.

Tests for the format_tool_error function to ensure:
- Returns valid JSON
- Correct structure {error_type: message}
- Default error_type is "error"
- Custom error_type works correctly
- Handles special characters in messages
"""

import json
import logging
from unittest.mock import patch

import pytest

from utils.tool_utils import format_tool_error, log_tool_error

pytestmark = pytest.mark.unit


class TestFormatToolError:
    """Tests for format_tool_error function."""

    def test_returns_valid_json(self):
        """Test that format_tool_error returns valid JSON."""
        result = format_tool_error("Test error message")
        # Should not raise an exception
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_correct_structure_with_default_error_type(self):
        """Test that the structure is {error_type: message} with default error_type."""
        message = "Test error message"
        result = format_tool_error(message)
        parsed = json.loads(result)

        assert "error" in parsed
        assert parsed["error"] == message
        assert len(parsed) == 1  # Should only have one key

    def test_default_error_type_is_error(self):
        """Test that default error_type is 'error'."""
        message = "Test error message"
        result = format_tool_error(message)
        parsed = json.loads(result)

        assert "error" in parsed
        assert parsed["error"] == message

    def test_custom_error_type_works_correctly(self):
        """Test that custom error_type works correctly."""
        message = "Test error message"
        custom_type = "validation_error"
        result = format_tool_error(message, error_type=custom_type)
        parsed = json.loads(result)

        assert custom_type in parsed
        assert parsed[custom_type] == message
        assert "error" not in parsed  # Should not have default error key

    def test_handles_special_characters_in_messages(self):
        """Test that special characters in messages are handled correctly."""
        special_chars = [
            'Error with quotes: "test"',
            "Error with newline:\nline 2",
            "Error with tab:\tindented",
            "Error with unicode: 🚀",
            "Error with backslash: \\test",
            "Error with JSON-like: {key: value}",
            "Error with brackets: [item1, item2]",
        ]

        for message in special_chars:
            result = format_tool_error(message)
            # Should not raise an exception
            parsed = json.loads(result)
            assert "error" in parsed
            assert parsed["error"] == message

    def test_handles_empty_message(self):
        """Test that empty message is handled correctly."""
        result = format_tool_error("")
        parsed = json.loads(result)

        assert "error" in parsed
        assert parsed["error"] == ""

    def test_handles_unicode_characters(self):
        """Test that unicode characters are handled correctly."""
        unicode_message = "Error: 测试 🚀 émojis"
        result = format_tool_error(unicode_message)
        parsed = json.loads(result)

        assert "error" in parsed
        assert parsed["error"] == unicode_message

    def test_handles_multiline_message(self):
        """Test that multiline messages are handled correctly."""
        multiline_message = "Line 1\nLine 2\nLine 3"
        result = format_tool_error(multiline_message)
        parsed = json.loads(result)

        assert "error" in parsed
        assert parsed["error"] == multiline_message

    def test_custom_error_type_with_special_characters(self):
        """Test custom error_type with special characters in message."""
        message = 'Error: "quotes" and {braces}'
        custom_type = "custom_error"
        result = format_tool_error(message, error_type=custom_type)
        parsed = json.loads(result)

        assert custom_type in parsed
        assert parsed[custom_type] == message

    def test_json_indentation(self):
        """Test that JSON output has proper indentation."""
        result = format_tool_error("Test message")
        # Should have indentation (indent=2)
        assert "\n" in result
        lines = result.split("\n")
        # Should have multiple lines due to indentation
        assert len(lines) > 1


class TestLogToolError:
    """Tests for log_tool_error function."""

    def test_returns_same_as_format_tool_error(self):
        """log_tool_error return value equals format_tool_error(message)."""
        logger = logging.getLogger("test_log_tool_error")
        message = "Something went wrong"
        result = log_tool_error(logger, message)
        assert result == format_tool_error(message)

    def test_logs_exception(self):
        """log_tool_error calls logger.exception with the message."""
        logger = logging.getLogger("test_log_tool_error_logs")
        message = "Tool failed"
        exc_calls = []
        with patch.object(
            logger, "exception", side_effect=lambda msg: exc_calls.append(msg)
        ):
            log_tool_error(logger, message)
        assert exc_calls == [message]

    def test_custom_error_type(self):
        """log_tool_error passes error_type to format_tool_error."""
        logger = logging.getLogger("test_log_tool_error_type")
        message = "Validation failed"
        custom_type = "validation_error"
        result = log_tool_error(logger, message, error_type=custom_type)
        assert result == format_tool_error(message, error_type=custom_type)
        parsed = json.loads(result)
        assert custom_type in parsed
        assert parsed[custom_type] == message
