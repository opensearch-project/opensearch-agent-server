"""Unit tests for the multi-template fill strategy."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agents.agentic_search.strategies.base import GenerationRequest
from agents.agentic_search.strategies.multi_template_fill import (
    MultiTemplateFillStrategy,
    _param_json_schema,
    _prefix_for,
    _prefixes_for,
)
from agents.agentic_search.template_schema import TemplateSchema, build_fill_model

pytestmark = pytest.mark.unit


class _FakeClient:
    """Minimal stand-in for the OpenSearch client used by the fill path."""

    def __init__(
        self,
        rendered: dict[str, Any] | None = None,
        *,
        mapping_error: Exception | None = None,
    ) -> None:
        self.rendered = (
            rendered if rendered is not None else {"query": {"match_all": {}}}
        )
        self.render_calls: list[tuple[str, dict[str, Any]]] = []
        # Per-instance so a test can observe the fallback's mapping fetch, and make it fail.
        self.indices = _FakeIndices(mapping_error)

    def render_search_template(
        self, *, id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        self.render_calls.append((id, body["params"]))
        return {"template_output": self.rendered}


class _FakeIndices:
    """Records mapping fetches and can be made to fail, like a real cluster."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    def get_mapping(self, index: str) -> dict[str, Any]:
        self.calls.append(index)
        if self.error is not None:
            raise self.error
        return {index: {"mappings": {}}}


class _FakeCache:
    """Serves prepared schemas; raises for ids marked unresolvable."""

    def __init__(
        self, schemas: dict[str, TemplateSchema], missing: set[str] | None = None
    ):
        self._schemas = schemas
        self._missing = missing or set()
        self.requested: list[str] = []

    def get(self, template_id: str, client: Any) -> TemplateSchema:
        self.requested.append(template_id)
        if template_id in self._missing:
            raise ValueError(f"no param-schema registered for '{template_id}'")
        return self._schemas[template_id]


class _RecordingSingle:
    """Captures the request the single-template path was handed."""

    name = "template_fill"
    needs_mapping = False

    def __init__(self) -> None:
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        self.calls.append(request)
        return {"query": {"term": {"single": True}}}


class _RecordingFallback:
    name = "direct_dsl"
    needs_mapping = True

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        self.calls += 1
        return {"query": {"match_all": {}}}


def _schema(
    tid: str, index: str, params: dict[str, Any], desc: str = ""
) -> TemplateSchema:
    return TemplateSchema(
        template_id=tid,
        index_binding=index,
        param_schema=params,
        fill_model=build_fill_model(params),
        description=desc,
    )


CATALOG = {
    "lex_query": {"type": "string", "description": "text"},
    "price_max": {"type": "number", "description": "max price"},
}
LISTINGS = {
    "lex_query": {"type": "string", "description": "text"},
    "beds_min": {"type": "number", "description": "min beds"},
}


def _request(
    context: dict[str, Any], client: Any, model: Any = None
) -> GenerationRequest:
    return GenerationRequest(
        question="cheap tents",
        index_name="idx-a",
        mapping="",
        context=context,
        model=model,
        client=client,
    )


class _FakeModel:
    """Bedrock-shaped model whose forced tool call returns a canned payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.config = {"model_id": "m", "temperature": 0}
        self.sent: list[dict[str, Any]] = []
        outer = self

        class _Client:
            def converse_stream(self, **kwargs):
                outer.sent.append(kwargs)
                return {
                    "stream": [
                        {
                            "contentBlockDelta": {
                                "delta": {
                                    "toolUse": {"input": json.dumps(outer._payload)}
                                }
                            }
                        },
                    ]
                }

        self.client = _Client()


# ---- single-candidate delegation -----------------------------------------


def test_single_surviving_candidate_delegates_to_single_template_path():
    """One candidate needs no choice, so the focused fill prompt should be used."""
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cache = _FakeCache({"t1": _schema("t1", "idx-a", CATALOG)})
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache
    )

    out = strat.generate(_request({"template_ids": ["t1"]}, _FakeClient()))

    assert out == {"query": {"term": {"single": True}}}
    assert fallback.calls == 0
    assert len(single.calls) == 1
    ctx = single.calls[0].context
    assert ctx["template_id"] == "t1"
    assert "template_ids" not in ctx


def test_scalar_and_list_naming_the_same_template_is_one_candidate():
    """The shared id parser dedupes, so this must still take the single-template path."""
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cache = _FakeCache({"t1": _schema("t1", "idx-a", CATALOG)})
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache
    )

    strat.generate(
        _request({"template_ids": ["t1"], "template_id": "t1"}, _FakeClient())
    )

    assert len(single.calls) == 1
    assert single.calls[0].context["template_id"] == "t1"
    assert fallback.calls == 0


def test_index_filter_reducing_to_one_candidate_delegates():
    """Candidates bound to another index are dropped before the choice is made."""
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cache = _FakeCache(
        {
            "t1": _schema("t1", "idx-a", CATALOG),
            "t2": _schema("t2", "idx-OTHER", LISTINGS),
        }
    )
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache
    )

    strat.generate(_request({"template_ids": ["t1", "t2"]}, _FakeClient()))

    assert len(single.calls) == 1
    assert single.calls[0].context["template_id"] == "t1"


def test_unresolvable_candidate_is_skipped_not_fatal():
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cache = _FakeCache({"good": _schema("good", "idx-a", CATALOG)}, missing={"bad"})
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache
    )

    strat.generate(_request({"template_ids": ["bad", "good"]}, _FakeClient()))

    assert len(single.calls) == 1
    assert single.calls[0].context["template_id"] == "good"
    assert fallback.calls == 0


def test_no_resolvable_candidate_falls_back():
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cache = _FakeCache({}, missing={"a", "b"})
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache
    )

    strat.generate(_request({"template_ids": ["a", "b"]}, _FakeClient()))

    assert fallback.calls == 1
    assert single.calls == []


def test_no_candidates_falls_back():
    single, fallback = _RecordingSingle(), _RecordingFallback()
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=_FakeCache({})
    )
    strat.generate(_request({}, _FakeClient()))
    assert fallback.calls == 1


# ---- the combined call ---------------------------------------------------


def _two_candidate_strategy(payload: dict[str, Any], **kw):
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cache = _FakeCache(
        {
            "t1": _schema("t1", "idx-a", CATALOG, "catalog"),
            "t2": _schema("t2", "idx-a", LISTINGS, "listings"),
        }
    )
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache, **kw
    )
    client = _FakeClient({"query": {"match": {"title": "tents"}}})
    request = _request({"template_ids": ["t1", "t2"]}, client, _FakeModel(payload))
    return strat, request, client, single, fallback


def test_combined_call_picks_and_renders_chosen_template():
    strat, request, client, single, fallback = _two_candidate_strategy(
        {"template_id": "t1", "t1__lex_query": "tents", "t1__price_max": 200}
    )

    out = strat.generate(request)

    assert out == {"query": {"match": {"title": "tents"}}}
    # Rendered against the chosen template with prefixes stripped.
    assert client.render_calls == [("t1", {"lex_query": "tents", "price_max": 200})]
    assert single.calls == []
    assert fallback.calls == 0


def test_combined_call_ignores_values_for_unchosen_template():
    strat, request, client, _, _ = _two_candidate_strategy(
        {"template_id": "t1", "t1__lex_query": "tents", "t2__beds_min": 3}
    )
    strat.generate(request)
    assert client.render_calls == [("t1", {"lex_query": "tents"})]


def test_cannot_express_routes_to_fallback():
    strat, request, client, single, fallback = _two_candidate_strategy(
        {"cannot_express": True, "template_id": "t1", "t1__lex_query": "x"}
    )
    strat.generate(request)
    assert fallback.calls == 1
    assert client.render_calls == []


def test_none_choice_routes_to_fallback():
    strat, request, client, _, fallback = _two_candidate_strategy(
        {"template_id": "none"}
    )
    strat.generate(request)
    assert fallback.calls == 1
    assert client.render_calls == []


def test_unknown_choice_routes_to_fallback():
    strat, request, client, _, fallback = _two_candidate_strategy(
        {"template_id": "nope"}
    )
    strat.generate(request)
    assert fallback.calls == 1
    assert client.render_calls == []


def test_candidate_set_is_capped():
    single, fallback = _RecordingSingle(), _RecordingFallback()
    schemas = {f"t{i}": _schema(f"t{i}", "idx-a", CATALOG) for i in range(6)}
    cache = _FakeCache(schemas)
    payload = {"template_id": "t0", "t0__lex_query": "x"}
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache, max_candidates=2
    )
    client, model = _FakeClient(), _FakeModel(payload)
    strat.generate(_request({"template_ids": list(schemas)}, client, model))

    # Assert on the schema the strategy actually sent, not one rebuilt by the test.
    sent = model.sent[0]["toolConfig"]["tools"][0]["toolSpec"]
    assert sent["inputSchema"]["json"]["properties"]["template_id"]["enum"] == [
        "t0",
        "t1",
        "none",
    ]
    # Work is bounded too: ids past the cap are never fetched.
    assert cache.requested == ["t0", "t1"]


def test_tool_schema_declares_abstain_before_choice_and_namespaces_params():
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cands = [
        _schema("t1", "idx-a", CATALOG, "catalog"),
        _schema("t2", "idx-a", LISTINGS, "listings"),
    ]
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=_FakeCache({})
    )
    spec, name_map = strat._build_tool_spec(cands)
    props = spec["inputSchema"]["json"]["properties"]

    # Abstain is offered first so expressibility is decided before committing.
    assert list(props)[:2] == ["cannot_express", "template_id"]
    assert spec["inputSchema"]["json"]["required"] == ["template_id"]
    # Same-named slots from different candidates stay separate.
    assert "t1__lex_query" in props and "t2__lex_query" in props
    assert props["t1__price_max"]["type"] == "number"


def test_render_failure_falls_back():
    class _Broken(_FakeClient):
        def render_search_template(self, *, id, body):
            return {"no_output": True}

    single, fallback = _RecordingSingle(), _RecordingFallback()
    cache = _FakeCache(
        {"t1": _schema("t1", "idx-a", CATALOG), "t2": _schema("t2", "idx-a", LISTINGS)}
    )
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache
    )
    model = _FakeModel({"template_id": "t1", "t1__lex_query": "x"})
    strat.generate(_request({"template_ids": ["t1", "t2"]}, _Broken(), model))
    assert fallback.calls == 1


# ---- schema helpers ------------------------------------------------------


def test_prefix_sanitizes_template_id():
    assert _prefix_for("a-b.c") == "a_b_c__"


def test_enum_param_becomes_string_enum():
    out = _param_json_schema({"type": "string", "enum": ["a", "b"], "description": "d"})
    assert out == {"description": "d", "type": "string", "enum": ["a", "b"]}


def test_array_param_is_string_with_literal_hint():
    out = _param_json_schema({"type": "array", "description": "values"})
    assert out["type"] == "string"
    assert "JSON array literal" in out["description"]


def test_colliding_template_ids_get_distinct_prefixes():
    """Ids that sanitize identically must not share a namespace."""
    prefixes = _prefixes_for(["a-b", "a_b"])
    assert prefixes["a-b"] != prefixes["a_b"]
    assert len(set(prefixes.values())) == 2


def test_colliding_ids_keep_separate_param_schemas():
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cands = [_schema("a-b", "idx-a", CATALOG), _schema("a_b", "idx-a", LISTINGS)]
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=_FakeCache({})
    )
    spec, name_map = strat._build_tool_spec(cands)
    # Each template keeps every one of its own params.
    assert set(name_map["a-b"].values()) == set(CATALOG)
    assert set(name_map["a_b"].values()) == set(LISTINGS)
    # And no emitted key is shared between them.
    assert not (set(name_map["a-b"]) & set(name_map["a_b"]))
    props = spec["inputSchema"]["json"]["properties"]
    assert all(k in props for k in name_map["a-b"])
    assert all(k in props for k in name_map["a_b"])


def test_nested_prefix_ids_do_not_leak_params():
    """With ids "x" and "x__y", x's prefix is a prefix of x__y's."""
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cache = _FakeCache(
        {
            "x": _schema("x", "idx-a", CATALOG),
            "x__y": _schema("x__y", "idx-a", LISTINGS),
        }
    )
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache
    )
    _, name_map = strat._build_tool_spec(
        [cache.get("x", None), cache.get("x__y", None)]
    )
    sibling_key = next(iter(name_map["x__y"]))
    # The sibling's key must not be readable as one of x's own params.
    assert sibling_key not in name_map["x"]

    payload = {"template_id": "x", "x__lex_query": "tents", sibling_key: "leak"}
    client = _FakeClient()
    strat.generate(
        _request({"template_ids": ["x", "x__y"]}, client, _FakeModel(payload))
    )
    assert client.render_calls == [("x", {"lex_query": "tents"})]


def test_string_false_does_not_abstain():
    """The raw tool input is unvalidated JSON, so "false" must not read as true."""
    strat, request, client, _, fallback = _two_candidate_strategy(
        {"cannot_express": "false", "template_id": "t1", "t1__lex_query": "tents"}
    )
    strat.generate(request)
    assert fallback.calls == 0
    assert client.render_calls == [("t1", {"lex_query": "tents"})]


def test_string_true_does_abstain():
    strat, request, client, _, fallback = _two_candidate_strategy(
        {"cannot_express": "true", "template_id": "t1"}
    )
    strat.generate(request)
    assert fallback.calls == 1
    assert client.render_calls == []


def test_numeric_enum_keeps_its_declared_type():
    """A numeric enum declared as string would emit "5" and fail Literal validation."""
    out = _param_json_schema({"type": "number", "enum": [1, 5, 10]})
    assert out["enum"] == [1, 5, 10]
    assert out["type"] == "number"


def test_candidate_named_none_is_still_selectable():
    """A template literally called "none" must not be swallowed by the sentinel."""
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cache = _FakeCache(
        {
            "none": _schema("none", "idx-a", CATALOG),
            "t2": _schema("t2", "idx-a", LISTINGS),
        }
    )
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache
    )
    spec, _ = strat._build_tool_spec([cache.get("none", None), cache.get("t2", None)])
    enum = spec["inputSchema"]["json"]["properties"]["template_id"]["enum"]
    # No duplicated member (invalid JSON Schema).
    assert len(enum) == len(set(enum))

    client, model = (
        _FakeClient(),
        _FakeModel({"template_id": "none", "none__lex_query": "x"}),
    )
    strat.generate(_request({"template_ids": ["none", "t2"]}, client, model))
    assert client.render_calls == [("none", {"lex_query": "x"})]
    assert fallback.calls == 0


def test_fallback_refetches_mapping_the_happy_path_skipped():
    """needs_mapping is False, so the fallback must fetch the mapping itself."""
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cache = _FakeCache({}, missing={"a", "b"})
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache
    )
    client = _FakeClient()

    strat.generate(_request({"template_ids": ["a", "b"]}, client))

    assert client.indices.calls == ["idx-a"]
    assert fallback.calls == 1


def test_fallback_still_runs_when_mapping_fetch_fails():
    single, fallback = _RecordingSingle(), _RecordingFallback()
    cache = _FakeCache({}, missing={"a", "b"})
    strat = MultiTemplateFillStrategy(
        single=single, fallback=fallback, schema_cache=cache
    )
    client = _FakeClient(mapping_error=RuntimeError("cluster down"))

    out = strat.generate(_request({"template_ids": ["a", "b"]}, client))

    assert client.indices.calls == ["idx-a"]
    assert fallback.calls == 1
    assert out == {"query": {"match_all": {}}}
