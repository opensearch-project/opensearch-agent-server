"""Unit tests for the template_fill generation strategy.

The Bedrock fill and the OpenSearch client are stubbed, so nothing hits a cluster
or an LLM. Covers: the happy path (fill -> _render/template -> unwrapped DSL), the
mode switch + strategy selection in the agent, and every degrade-to-free-DSL
fallback branch (no template_id, unregistered template, fill failure, render
failure), including that the fallback re-fetches the mapping the happy path skips.
"""

from __future__ import annotations

import json

import pytest

from agents.agentic_search import agent as agent_module
from agents.agentic_search.agent import AgenticSearchAgent
from agents.agentic_search.strategies import template_fill as tf_module
from agents.agentic_search.strategies.base import GenerationRequest
from agents.agentic_search.strategies.template_fill import TemplateFillStrategy
from agents.agentic_search.template_schema import TemplateSchema, build_fill_model

pytestmark = pytest.mark.unit

PRODUCT_SCHEMA = {
    "lex_query": {"type": "string", "required": True},
    "size": {"type": "integer", "required": False},
    "color": {"type": "string", "required": False},
    "price_max": {"type": "float", "required": False},
}

RENDERED_DSL = {
    "size": 5,
    "query": {
        "bool": {"must": [{"multi_match": {"query": "shoes", "fields": ["title"]}}]}
    },
}


class _StubSchemaCache:
    """Returns a prebuilt TemplateSchema (or raises to simulate unregistered)."""

    def __init__(self, *, schema=None, raises=None):
        self._schema = schema
        self._raises = raises
        self.calls = []

    def get(self, template_id, client):
        self.calls.append(template_id)
        if self._raises is not None:
            raise self._raises
        return self._schema


class _StubFallback:
    """Free-DSL fallback stub; records the request it was handed."""

    name = "direct_dsl"
    needs_mapping = True

    def __init__(self, returns=None):
        self._returns = returns if returns is not None else {"query": {"match_all": {}}}
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return self._returns


class _StubRenderClient:
    """Stands in for the OpenSearch client's render_search_template + get_mapping."""

    def __init__(self, *, template_output=RENDERED_DSL, render_exc=None, mapping=None):
        self._template_output = template_output
        self._render_exc = render_exc
        self._mapping = (
            mapping if mapping is not None else {"products": {"mappings": {}}}
        )
        self.render_calls = []
        self.mapping_calls = []

    def render_search_template(self, id, body):  # noqa: A002 - mirrors client sig
        self.render_calls.append((id, body))
        if self._render_exc is not None:
            raise self._render_exc
        return {"template_output": self._template_output}

    class _Indices:
        def __init__(self, outer):
            self._outer = outer

        def get_mapping(self, index):
            self._outer.mapping_calls.append(index)
            return self._outer._mapping

    @property
    def indices(self):
        return _StubRenderClient._Indices(self)


def _make_request(client, *, template_id="product_search", mapping="", model=None):
    return GenerationRequest(
        question="5 cheapest red shoes under $100",
        index_name="products",
        mapping=mapping,
        context={"index_name": "products", "template_id": template_id}
        if template_id
        else {"index_name": "products"},
        model=model or object(),
        client=client,
    )


def _product_schema_obj():
    return TemplateSchema(
        template_id="product_search",
        index_binding="products",
        param_schema=PRODUCT_SCHEMA,
        fill_model=build_fill_model(PRODUCT_SCHEMA),
    )


# --- happy path -------------------------------------------------------------


def test_happy_path_fills_renders_and_unwraps(monkeypatch):
    client = _StubRenderClient()
    cache = _StubSchemaCache(schema=_product_schema_obj())
    strat = TemplateFillStrategy(schema_cache=cache)

    # Stub the forced fill to return validated params (no Bedrock).
    filled = _product_schema_obj().fill_model.model_validate(
        {"lex_query": "shoes", "size": 5, "color": "red", "price_max": 100}
    )
    monkeypatch.setattr(tf_module, "forced_tool_fill", lambda **kw: filled)

    out = strat.generate(_make_request(client))

    assert out == RENDERED_DSL
    # Rendered by template_id with only the filled (non-None) params.
    assert client.render_calls == [
        (
            "product_search",
            {
                "params": {
                    "lex_query": "shoes",
                    "size": 5,
                    "color": "red",
                    "price_max": 100.0,
                }
            },
        )
    ]
    assert cache.calls == ["product_search"]


def test_cannot_express_routes_to_fallback(monkeypatch):
    # The model sets the abstain field -> skip render, route to free-DSL. The abstain
    # field must never reach the renderer.
    fb = _StubFallback()
    client = _StubRenderClient()
    strat = TemplateFillStrategy(
        fallback=fb, schema_cache=_StubSchemaCache(schema=_product_schema_obj())
    )
    filled = _product_schema_obj().fill_model.model_validate(
        {"lex_query": "x", "cannot_express": True}
    )
    monkeypatch.setattr(tf_module, "forced_tool_fill", lambda **kw: filled)

    out = strat.generate(_make_request(client, mapping='{"m":1}'))

    assert out == {"query": {"match_all": {}}}  # the fallback's result
    assert len(fb.calls) == 1
    assert client.render_calls == []  # never rendered


def test_cannot_express_false_renders_normally(monkeypatch):
    # abstain=False is the happy path: the synthetic field is stripped, render proceeds.
    client = _StubRenderClient()
    strat = TemplateFillStrategy(
        schema_cache=_StubSchemaCache(schema=_product_schema_obj())
    )
    filled = _product_schema_obj().fill_model.model_validate(
        {"lex_query": "shoes", "cannot_express": False}
    )
    monkeypatch.setattr(tf_module, "forced_tool_fill", lambda **kw: filled)

    out = strat.generate(_make_request(client))

    assert out == RENDERED_DSL
    # cannot_express is stripped before the params reach the renderer.
    assert client.render_calls == [
        ("product_search", {"params": {"lex_query": "shoes"}})
    ]


def test_render_output_string_is_parsed(monkeypatch):
    # Some client versions return template_output as a JSON string.
    client = _StubRenderClient(template_output=json.dumps(RENDERED_DSL))
    strat = TemplateFillStrategy(
        schema_cache=_StubSchemaCache(schema=_product_schema_obj())
    )
    filled = _product_schema_obj().fill_model.model_validate({"lex_query": "shoes"})
    monkeypatch.setattr(tf_module, "forced_tool_fill", lambda **kw: filled)

    out = strat.generate(_make_request(client))
    assert out == RENDERED_DSL


# --- fallback branches ------------------------------------------------------


def test_no_template_id_falls_back():
    fb = _StubFallback()
    strat = TemplateFillStrategy(fallback=fb, schema_cache=_StubSchemaCache())
    client = _StubRenderClient()
    out = strat.generate(_make_request(client, template_id=None, mapping='{"m":1}'))
    assert out == {"query": {"match_all": {}}}
    assert len(fb.calls) == 1
    assert client.render_calls == []  # never attempted a render


def test_unregistered_template_falls_back():
    fb = _StubFallback()
    cache = _StubSchemaCache(raises=ValueError("no param-schema registered"))
    strat = TemplateFillStrategy(fallback=fb, schema_cache=cache)
    out = strat.generate(_make_request(_StubRenderClient(), mapping='{"m":1}'))
    assert out == {"query": {"match_all": {}}}
    assert len(fb.calls) == 1


def test_fill_failure_falls_back(monkeypatch):
    fb = _StubFallback()
    strat = TemplateFillStrategy(
        fallback=fb, schema_cache=_StubSchemaCache(schema=_product_schema_obj())
    )

    def _boom(**kw):
        raise ValueError("forced tool input failed validation")

    monkeypatch.setattr(tf_module, "forced_tool_fill", _boom)
    out = strat.generate(_make_request(_StubRenderClient(), mapping='{"m":1}'))
    assert out == {"query": {"match_all": {}}}
    assert len(fb.calls) == 1


def test_render_failure_falls_back(monkeypatch):
    fb = _StubFallback()
    client = _StubRenderClient(render_exc=RuntimeError("bad Mustache"))
    strat = TemplateFillStrategy(
        fallback=fb, schema_cache=_StubSchemaCache(schema=_product_schema_obj())
    )
    filled = _product_schema_obj().fill_model.model_validate({"lex_query": "shoes"})
    monkeypatch.setattr(tf_module, "forced_tool_fill", lambda **kw: filled)

    out = strat.generate(_make_request(client, mapping='{"m":1}'))
    assert out == {"query": {"match_all": {}}}
    assert len(fb.calls) == 1


def test_missing_template_output_falls_back(monkeypatch):
    fb = _StubFallback()
    client = _StubRenderClient(template_output=None)  # -> render returns no output
    # override render to return an empty dict (no template_output key)
    client.render_search_template = lambda id, body: {}  # noqa: A002
    strat = TemplateFillStrategy(
        fallback=fb, schema_cache=_StubSchemaCache(schema=_product_schema_obj())
    )
    filled = _product_schema_obj().fill_model.model_validate({"lex_query": "shoes"})
    monkeypatch.setattr(tf_module, "forced_tool_fill", lambda **kw: filled)

    out = strat.generate(_make_request(client, mapping='{"m":1}'))
    assert out == {"query": {"match_all": {}}}


def test_fallback_refetches_mapping_when_absent():
    # Happy path skips the mapping (needs_mapping=False), so the fallback must fetch
    # it. The stub fallback records the request it received; assert it has a mapping.
    fb = _StubFallback()
    client = _StubRenderClient()
    cache = _StubSchemaCache(raises=ValueError("unregistered"))
    strat = TemplateFillStrategy(fallback=fb, schema_cache=cache)

    strat.generate(_make_request(client, mapping=""))  # no mapping on the request

    assert client.mapping_calls == ["products"]  # fallback fetched it
    handed = fb.calls[0]
    assert json.loads(handed.mapping) == {"products": {"mappings": {}}}


# --- agent-level mode switch ------------------------------------------------


class _AgentStubStrategy:
    name = "template_fill"
    needs_mapping = False

    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return {"query": {"term": {"color": "red"}}}


def test_agent_auto_selects_template_fill_on_template_id(monkeypatch):
    """template_id present -> the agent routes to template_fill and skips mapping."""
    monkeypatch.setattr(agent_module, "create_model", lambda: object())
    strat = _AgentStubStrategy()
    monkeypatch.setattr(
        agent_module, "STRATEGIES", {"template_fill": strat, "direct_dsl": object()}
    )
    monkeypatch.setattr(agent_module, "DEFAULT_STRATEGY", "direct_dsl")

    a = AgenticSearchAgent("http://localhost:9200")
    client = _StubRenderClient()
    monkeypatch.setattr(a, "_client", lambda auth_token: client)

    out = a(
        "red shoes", context={"index_name": "products", "template_id": "product_search"}
    )

    assert json.loads(out) == {"query": {"term": {"color": "red"}}}
    assert len(strat.calls) == 1
    # needs_mapping=False -> the agent did NOT fetch the mapping upfront.
    assert client.mapping_calls == []
    assert strat.calls[0].mapping == ""


def test_agent_defaults_to_free_dsl_without_template_id(monkeypatch):
    """No template_id -> the default (free-DSL) strategy, mapping fetched."""
    monkeypatch.setattr(agent_module, "create_model", lambda: object())

    class _Default:
        name = "direct_dsl"
        needs_mapping = True

        def __init__(self):
            self.calls = []

        def generate(self, request):
            self.calls.append(request)
            return {"query": {"match_all": {}}}

    default = _Default()
    monkeypatch.setattr(agent_module, "STRATEGIES", {"direct_dsl": default})
    monkeypatch.setattr(agent_module, "DEFAULT_STRATEGY", "direct_dsl")

    a = AgenticSearchAgent("http://localhost:9200")
    client = _StubRenderClient()
    monkeypatch.setattr(a, "_client", lambda auth_token: client)

    a("red shoes", context={"index_name": "products"})

    assert len(default.calls) == 1
    assert client.mapping_calls == ["products"]  # mapping fetched for free-DSL
