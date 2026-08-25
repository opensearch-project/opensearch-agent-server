# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Coverage tests for server.route_helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import server.route_helpers as rh
from server.route_helpers import (
    create_encoder,
    ensure_thread_has_title,
    generate_thread_title_from_message,
    save_initial_messages,
)

pytestmark = pytest.mark.unit


class TestCreateEncoder:
    def test_returns_event_encoder_when_available(self):
        """With ag-ui installed, a real EventEncoder is returned."""
        enc = create_encoder("text/event-stream")
        assert enc is not None
        assert hasattr(enc, "encode")

    def test_fallback_simple_encoder(self, monkeypatch):
        """When EventEncoder is unavailable, the SimpleEncoder fallback is used."""
        monkeypatch.setattr(rh, "HAS_EVENT_ENCODER", False)
        enc = create_encoder("text/event-stream")
        assert enc.get_content_type() == "text/event-stream"

        # Pydantic-v2 model_dump_json path
        model = MagicMock()
        model.model_dump_json.return_value = '{"a":1}'
        # ensure model_dump is not preferred over model_dump_json
        out = enc.encode(model)
        assert out == 'data: {"a":1}\n\n'

        # model_dump path (no model_dump_json)
        class OnlyDump:
            def model_dump(self, **kwargs):
                return {"b": 2}

        out2 = enc.encode(OnlyDump())
        assert out2 == f"data: {json.dumps({'b': 2})}\n\n"

        # plain dict path
        out3 = enc.encode({"c": 3})
        assert out3 == f"data: {json.dumps({'c': 3})}\n\n"


class TestGenerateThreadTitle:
    def test_empty_returns_new_chat(self):
        assert generate_thread_title_from_message("") == "New Chat"

    def test_plain_string(self):
        assert generate_thread_title_from_message("Hello world") == "Hello world"

    def test_collapses_whitespace_and_newlines(self):
        assert generate_thread_title_from_message("a\n\n  b   c") == "a b c"

    def test_list_content_with_text_items(self):
        content = [{"text": "find"}, {"type": "text", "text": "logs"}, "now", 5]
        assert generate_thread_title_from_message(content) == "find logs now"

    def test_dict_content(self):
        assert generate_thread_title_from_message({"text": "hi there"}) == "hi there"

    def test_dict_content_uses_content_key(self):
        assert generate_thread_title_from_message({"content": "howdy"}) == "howdy"

    def test_word_boundary_truncation(self):
        msg = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
        title = generate_thread_title_from_message(msg, max_length=20)
        assert title.endswith("...")
        assert len(title) <= 20

    def test_long_single_word_truncation(self):
        # A single long word (no spaces near cut) forces simple truncation.
        msg = "x" * 100
        title = generate_thread_title_from_message(msg, max_length=20)
        assert title.endswith("...")
        assert len(title) <= 20

    def test_list_without_text_items_stringifies(self):
        # A list containing no extractable text falls back to str(content).
        title = generate_thread_title_from_message([123, {"other": 1}])
        assert title  # non-empty string representation


class TestSaveInitialMessages:
    def test_no_persistence_is_noop(self):
        # Should simply return without error.
        save_initial_messages(
            None, SimpleNamespace(messages=[{"role": "user"}]), "t", "r"
        )

    def test_no_messages_attr_is_noop(self):
        persistence = MagicMock()
        save_initial_messages(persistence, SimpleNamespace(), "t", "r")
        persistence.save_message.assert_not_called()

    def test_saves_new_user_messages_dict_and_object(self):
        persistence = MagicMock()
        persistence.get_messages.return_value = []
        obj_msg = SimpleNamespace(role="user", content="from object")
        input_data = SimpleNamespace(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ignored"},
                obj_msg,
            ]
        )
        save_initial_messages(persistence, input_data, "t1", "r1")
        # Only the two user messages saved.
        assert persistence.save_message.call_count == 2

    def test_skips_duplicate_messages(self):
        persistence = MagicMock()
        persistence.get_messages.return_value = [{"role": "user", "content": "dup"}]
        input_data = SimpleNamespace(messages=[{"role": "user", "content": "dup"}])
        save_initial_messages(persistence, input_data, "t", "r")
        persistence.save_message.assert_not_called()

    def test_object_message_text_fallback(self):
        persistence = MagicMock()
        persistence.get_messages.return_value = []
        # content is None so the code falls back to `.text`
        msg = SimpleNamespace(role="user", content=None, text="via text attr")
        save_initial_messages(persistence, SimpleNamespace(messages=[msg]), "t", "r")
        assert persistence.save_message.call_count == 1

    def test_get_messages_exception_is_handled(self):
        persistence = MagicMock()
        persistence.get_messages.side_effect = RuntimeError("db down")
        input_data = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])
        # Should not raise; falls back to empty existing set and saves.
        save_initial_messages(persistence, input_data, "t", "r")
        assert persistence.save_message.call_count == 1


class TestEnsureThreadHasTitle:
    def test_no_persistence_is_noop(self):
        ensure_thread_has_title(None, "t", SimpleNamespace(messages=[]))

    def test_existing_title_returns_early(self):
        persistence = MagicMock()
        persistence.get_thread.return_value = {"metadata": {"title": "Existing"}}
        ensure_thread_has_title(persistence, "t", SimpleNamespace(messages=[]))
        persistence.save_thread.assert_not_called()

    def test_string_metadata_with_title_returns_early(self):
        persistence = MagicMock()
        persistence.get_thread.return_value = {"metadata": json.dumps({"title": "T"})}
        ensure_thread_has_title(persistence, "t", SimpleNamespace(messages=[]))
        persistence.save_thread.assert_not_called()

    def test_generates_title_from_first_user_message_no_metadata(self):
        persistence = MagicMock()
        persistence.get_thread.return_value = {"metadata": None}
        input_data = SimpleNamespace(
            messages=[
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "What is my index size?"},
            ]
        )
        ensure_thread_has_title(persistence, "t", input_data)
        persistence.save_thread.assert_called_once()
        _, kwargs = persistence.save_thread.call_args
        assert kwargs["metadata"]["title"] == "What is my index size?"

    def test_generates_title_merges_existing_string_metadata(self):
        persistence = MagicMock()
        # Metadata present as JSON string but WITHOUT a title -> generate + merge.
        persistence.get_thread.return_value = {"metadata": json.dumps({"pinned": True})}
        msg = SimpleNamespace(role="user", content=None, text="hello from text")
        ensure_thread_has_title(persistence, "t", SimpleNamespace(messages=[msg]))
        persistence.save_thread.assert_called_once()
        _, kwargs = persistence.save_thread.call_args
        assert kwargs["metadata"]["title"] == "hello from text"
        assert kwargs["metadata"]["pinned"] is True

    def test_no_thread_generates_from_message(self):
        persistence = MagicMock()
        persistence.get_thread.return_value = None
        input_data = SimpleNamespace(messages=[{"role": "user", "content": "boom"}])
        ensure_thread_has_title(persistence, "t", input_data)
        persistence.save_thread.assert_called_once()

    def test_invalid_json_string_metadata_generates_title(self):
        persistence = MagicMock()
        # Metadata is a non-JSON string -> parsed to {} -> title generated.
        persistence.get_thread.return_value = {"metadata": "not-json"}
        input_data = SimpleNamespace(messages=[{"role": "user", "content": "hey"}])
        ensure_thread_has_title(persistence, "t", input_data)
        persistence.save_thread.assert_called_once()

    def test_merges_existing_dict_metadata(self):
        persistence = MagicMock()
        # Metadata present as a dict WITHOUT a title -> generate + merge dict path.
        persistence.get_thread.return_value = {"metadata": {"pinned": True}}
        input_data = SimpleNamespace(messages=[{"role": "user", "content": "topic"}])
        ensure_thread_has_title(persistence, "t", input_data)
        persistence.save_thread.assert_called_once()
        _, kwargs = persistence.save_thread.call_args
        assert kwargs["metadata"]["title"] == "topic"
        assert kwargs["metadata"]["pinned"] is True

    def test_exception_is_swallowed(self):
        persistence = MagicMock()
        persistence.get_thread.side_effect = RuntimeError("boom")
        # Should not raise.
        ensure_thread_has_title(
            persistence,
            "t",
            SimpleNamespace(messages=[{"role": "user", "content": "x"}]),
        )
