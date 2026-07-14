"""Unit tests for the ``dsl_generator`` agent (reached via POST /invoke).

Verifies, with the generation engine mocked (no cluster, no LLM):
1. A successful generation returns the DSL string, and wrap_inference_results
   places it at output[0].result (so a connector's passthrough lands it in
   ModelTensor.result).
2. A generation error degrades to the fallback match_all DSL.
3. Output that isn't a JSON object is treated as failure -> fallback.
4. When no engine is registered, the fallback is returned.
5. The agent adapter maps (query, context.index_name, auth_token) onto the
   engine, and guards missing query / index_name with the fallback.
"""

from __future__ import annotations

import json

import pytest

from server import dsl_agent
from server.dsl_agent import (
    FALLBACK_DSL,
    DslInvokeAgent,
    generate_dsl_or_fallback,
    set_dsl_generator,
    wrap_inference_results,
)

pytestmark = pytest.mark.unit

MATCH_ALL = {"size": 10, "query": {"match_all": {}}}


class _StubGenerator:
    """Records the last generate() call and returns a canned value or raises."""

    def __init__(self, *, returns: str | None = None, raises: Exception | None = None):
        self._returns = returns
        self._raises = raises
        self.calls: list[tuple[str, str, str | None]] = []

    def generate(self, question, index_name, auth_token=None):
        self.calls.append((question, index_name, auth_token))
        if self._raises is not None:
            raise self._raises
        return self._returns


@pytest.fixture(autouse=True)
def _reset_generator():
    """Ensure each test starts and ends with no registered engine."""
    dsl_agent._generator = None
    yield
    dsl_agent._generator = None


# --- generate_dsl_or_fallback --------------------------------------------------


def test_success_returns_generated_dsl():
    dsl = '{"query":{"term":{"status":"active"}}}'
    gen = _StubGenerator(returns=dsl)
    set_dsl_generator(gen)

    result = generate_dsl_or_fallback("active items", "idx", auth_token="svc-token")

    assert result == dsl
    # Bearer token forwarded to the engine for cluster access.
    assert gen.calls == [("active items", "idx", "svc-token")]


def test_generator_exception_falls_back():
    set_dsl_generator(_StubGenerator(raises=RuntimeError("bedrock unavailable")))

    result = generate_dsl_or_fallback("x", "idx")

    assert result == FALLBACK_DSL
    assert json.loads(result) == MATCH_ALL


def test_non_json_output_falls_back():
    set_dsl_generator(_StubGenerator(returns="not valid json"))

    assert generate_dsl_or_fallback("x", "idx") == FALLBACK_DSL


@pytest.mark.parametrize("non_object", ['"ok"', "[]", "42", "true", "null"])
def test_non_object_json_falls_back(non_object):
    # Valid JSON that isn't an object is not a usable _search body -> fallback.
    set_dsl_generator(_StubGenerator(returns=non_object))

    assert generate_dsl_or_fallback("x", "idx") == FALLBACK_DSL


def test_no_generator_registered_falls_back():
    assert generate_dsl_or_fallback("x", "idx") == FALLBACK_DSL


# --- wrap_inference_results ----------------------------------------------------


def test_wrap_puts_dsl_string_at_output_result():
    dsl = '{"query":{"match_all":{}}}'
    env = wrap_inference_results(dsl)

    output = env["inference_results"][0]["output"][0]
    assert output["result"] == dsl
    # Must be a STRING at output[0].result, not a nested object.
    assert isinstance(output["result"], str)
    assert env["inference_results"][0]["status_code"] == 200


# --- DslInvokeAgent adapter ----------------------------------------------------


def test_agent_maps_query_and_context_onto_engine():
    dsl = '{"query":{"term":{"color":"red"}}}'
    gen = _StubGenerator(returns=dsl)
    set_dsl_generator(gen)

    result = DslInvokeAgent()(
        "red shoes", context={"index_name": "products"}, auth_token="tok"
    )

    assert result == dsl
    assert gen.calls == [("red shoes", "products", "tok")]


def test_agent_declares_context_awareness():
    # The orchestrator keys context/auth forwarding off this flag.
    assert DslInvokeAgent.accepts_invoke_context is True


def test_agent_without_index_name_falls_back():
    gen = _StubGenerator(returns='{"query":{"match_all":{}}}')
    set_dsl_generator(gen)

    result = DslInvokeAgent()("red shoes", context={})

    assert result == FALLBACK_DSL
    # Guarded before reaching the engine.
    assert gen.calls == []


def test_agent_without_query_falls_back():
    gen = _StubGenerator(returns='{"query":{"match_all":{}}}')
    set_dsl_generator(gen)

    result = DslInvokeAgent()("  ", context={"index_name": "products"})

    assert result == FALLBACK_DSL
    assert gen.calls == []
