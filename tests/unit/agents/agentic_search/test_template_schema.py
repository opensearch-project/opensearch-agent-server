"""Unit tests for the param-schema -> dynamic FillTemplate model + TTL cache.

Covers building the Pydantic model from a param-schema (types, enums, required vs
optional), the schema-doc fetch from the system index, and the TTL cache (hit,
miss, expiry, unregistered/empty-schema errors). No cluster or LLM is touched.
"""

from __future__ import annotations

import pytest

from agents.agentic_search import template_schema as ts_module
from agents.agentic_search.template_schema import (
    AGENTIC_SEARCH_TEMPLATES_INDEX,
    CANNOT_EXPRESS_FIELD,
    TemplateSchemaCache,
    build_fill_model,
)

pytestmark = pytest.mark.unit

# The worked-example schema from the design doc (§4.3).
PRODUCT_SCHEMA = {
    "lex_query": {
        "type": "string",
        "required": True,
        "description": "content words only",
    },
    "size": {"type": "integer", "required": False},
    "color": {"type": "string", "required": False},
    "brand": {"type": "string", "required": False},
    "price_max": {"type": "float", "required": False},
    "sort_by": {
        "type": "string",
        "enum": ["price", "rating", "created_at"],
        "required": False,
    },
    "sort_order": {"type": "string", "enum": ["asc", "desc"], "required": False},
}


class _StubClient:
    """Stands in for the OpenSearch client's ``get`` on the system index."""

    def __init__(self, doc=None, *, raise_exc=None):
        self._doc = doc
        self._raise = raise_exc
        self.get_calls = []

    def get(self, index, id):  # noqa: A002 - mirrors opensearch-py signature
        self.get_calls.append((index, id))
        if self._raise is not None:
            raise self._raise
        if self._doc is None:
            return {"found": False}
        return {"found": True, "_source": self._doc}


# --- build_fill_model -------------------------------------------------------


def test_build_model_required_and_optional():
    model = build_fill_model(PRODUCT_SCHEMA)
    schema = model.model_json_schema()
    # Only the required param is in `required`; optionals are nullable + default.
    assert schema["required"] == ["lex_query"]
    assert model.model_fields["size"].is_required() is False


def test_build_model_enum_becomes_literal():
    model = build_fill_model(PRODUCT_SCHEMA)
    # A value outside the enum is rejected; a valid one passes.
    with pytest.raises(Exception):
        model.model_validate({"lex_query": "x", "sort_by": "bogus"})
    inst = model.model_validate({"lex_query": "x", "sort_by": "price"})
    assert inst.sort_by == "price"


def test_build_model_ignores_extra_params():
    model = build_fill_model(PRODUCT_SCHEMA)
    # A hallucinated key is dropped (extra="ignore"), not rejected — one stray key
    # must not fail the whole fill; the render-parse guard backstops a bad result.
    inst = model.model_validate({"lex_query": "x", "not_a_param": 1})
    assert "not_a_param" not in inst.model_dump(by_alias=True, exclude_none=True)


def _params(inst, **dump_kwargs):
    """Dump a fill instance minus the synthetic abstain field (as the strategy does)."""
    dumped = inst.model_dump(by_alias=True, **dump_kwargs)
    dumped.pop(CANNOT_EXPRESS_FIELD, None)
    return dumped


def test_build_model_number_preserves_int():
    # A "number"-typed param filled with an integer must render as an int, not 5.0
    # (OpenSearch rejects a float where an int is expected, e.g. size).
    model = build_fill_model({"size": {"type": "number", "required": True}})
    dumped = _params(model.model_validate({"size": 5}))
    assert dumped == {"size": 5}
    assert isinstance(dumped["size"], int)
    # A decimal still stays a float.
    assert _params(model.model_validate({"size": 5.5})) == {"size": 5.5}


def test_build_model_coerces_and_types():
    model = build_fill_model(PRODUCT_SCHEMA)
    inst = model.model_validate({"lex_query": "shoes", "size": 5, "price_max": 100})
    dumped = _params(inst, exclude_none=True)
    # size is "integer" -> stays int; price_max is "float" -> 100.0.
    assert dumped == {"lex_query": "shoes", "size": 5, "price_max": 100.0}


def test_build_model_awkward_param_names():
    # Dotted, model_-prefixed, leading-underscore, and keyword names must not crash
    # create_model; the real name is preserved as the field alias.
    schema = {
        "author.first_name": {"type": "string", "required": True},
        "model_id": {"type": "string", "required": False},
        "_internal": {"type": "string", "required": False},
        "class": {"type": "string", "required": False},
    }
    model = build_fill_model(schema)
    inst = model.model_validate(
        {"author.first_name": "John", "model_id": "m1", "_internal": "z", "class": "c"}
    )
    dumped = _params(inst, exclude_none=True)
    assert dumped == {
        "author.first_name": "John",
        "model_id": "m1",
        "_internal": "z",
        "class": "c",
    }


def test_build_model_colliding_sanitized_names_stay_distinct():
    # "a.b" and "a-b" both sanitize to "a_b"; both must still round-trip by alias.
    schema = {
        "a.b": {"type": "string", "required": True},
        "a-b": {"type": "string", "required": True},
    }
    model = build_fill_model(schema)
    dumped = _params(model.model_validate({"a.b": "x", "a-b": "y"}))
    assert dumped == {"a.b": "x", "a-b": "y"}


def test_build_model_empty_schema_raises():
    with pytest.raises(ValueError, match="empty"):
        build_fill_model({})


def test_build_model_bad_enum_raises():
    with pytest.raises(ValueError, match="enum"):
        build_fill_model({"x": {"type": "string", "enum": []}})


def test_build_model_unknown_type_falls_back_to_string():
    model = build_fill_model({"weird": {"type": "geo_point", "required": True}})
    inst = model.model_validate({"weird": "anything"})
    assert inst.weird == "anything"


# --- cannot_express escape hatch --------------------------------------------


def test_build_model_adds_cannot_express_field():
    # The synthetic abstain field is added to every model, optional, default False,
    # and is not in `required` (so it costs nothing on the happy path).
    model = build_fill_model(PRODUCT_SCHEMA)
    assert CANNOT_EXPRESS_FIELD in model.model_fields
    assert CANNOT_EXPRESS_FIELD not in model.model_json_schema()["required"]
    inst = model.model_validate({"lex_query": "x"})
    assert getattr(inst, CANNOT_EXPRESS_FIELD) is False


def test_build_model_cannot_express_can_be_set():
    model = build_fill_model(PRODUCT_SCHEMA)
    inst = model.model_validate({"lex_query": "x", CANNOT_EXPRESS_FIELD: True})
    assert getattr(inst, CANNOT_EXPRESS_FIELD) is True
    # It round-trips by alias (plain identifier, so alias == field name).
    assert inst.model_dump(by_alias=True)[CANNOT_EXPRESS_FIELD] is True


def test_build_model_real_param_named_cannot_express_wins():
    # A template that literally names a param `cannot_express` keeps its real param
    # (the escape hatch is disabled for that template rather than shadowing the param).
    schema = {CANNOT_EXPRESS_FIELD: {"type": "string", "required": True}}
    model = build_fill_model(schema)
    # The field is the real (string) param, not the synthetic bool.
    inst = model.model_validate({CANNOT_EXPRESS_FIELD: "hello"})
    assert getattr(inst, CANNOT_EXPRESS_FIELD) == "hello"


# --- TemplateSchemaCache ----------------------------------------------------


def test_cache_fetches_and_builds():
    client = _StubClient(
        doc={"param_schema": PRODUCT_SCHEMA, "index_binding": "products"}
    )
    cache = TemplateSchemaCache()
    schema = cache.get("product_search", client)
    assert schema.template_id == "product_search"
    assert schema.index_binding == "products"
    assert client.get_calls == [(AGENTIC_SEARCH_TEMPLATES_INDEX, "product_search")]
    # The built model validates a fill.
    inst = schema.fill_model.model_validate({"lex_query": "shoes"})
    assert inst.lex_query == "shoes"


def test_cache_hit_avoids_second_fetch():
    client = _StubClient(doc={"param_schema": PRODUCT_SCHEMA})
    cache = TemplateSchemaCache(ttl_seconds=1000)
    cache.get("product_search", client)
    cache.get("product_search", client)
    assert len(client.get_calls) == 1  # second call served from cache


def test_cache_expiry_refetches(monkeypatch):
    client = _StubClient(doc={"param_schema": PRODUCT_SCHEMA})
    fake_now = {"t": 1000.0}
    monkeypatch.setattr(ts_module.time, "monotonic", lambda: fake_now["t"])
    cache = TemplateSchemaCache(ttl_seconds=60)
    cache.get("product_search", client)
    fake_now["t"] = 1000.0 + 61  # past the TTL
    cache.get("product_search", client)
    assert len(client.get_calls) == 2  # re-read after expiry


def test_cache_unregistered_template_raises():
    client = _StubClient(doc=None)  # {"found": False}
    cache = TemplateSchemaCache()
    with pytest.raises(ValueError, match="no param-schema registered"):
        cache.get("missing", client)


def test_cache_fetch_exception_treated_as_unregistered():
    client = _StubClient(raise_exc=RuntimeError("index_not_found_exception"))
    cache = TemplateSchemaCache()
    with pytest.raises(ValueError, match="no param-schema registered"):
        cache.get("missing", client)


def test_cache_empty_param_schema_raises():
    client = _StubClient(doc={"param_schema": {}})
    cache = TemplateSchemaCache()
    with pytest.raises(ValueError, match="missing or empty"):
        cache.get("empty", client)
