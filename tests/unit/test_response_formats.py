"""Unit tests for the /invoke response envelopes.

Pins the ml-commons ``inference_results`` shape: a connector's built-in
``mlcommons.passthrough`` copies ``output[0].result`` (a string) into
``ModelTensor.result``, which is the one field neural-search reads. This is the
cross-service contract, so the exact shape is asserted rather than trusted.
"""

from __future__ import annotations

import pytest

from server.response_formats import wrap_inference_results

pytestmark = pytest.mark.unit


def test_envelope_shape_puts_string_at_output_result():
    dsl = '{"query":{"match_all":{}}}'
    env = wrap_inference_results(dsl)

    # Exact contract shape consumed by connector.post_process.mlcommons.passthrough.
    assert list(env.keys()) == ["inference_results"]
    entry = env["inference_results"][0]
    assert entry["status_code"] == 200
    output = entry["output"][0]
    assert output["name"] == "response"
    # Must be a STRING at output[0].result, not a nested object.
    assert output["result"] == dsl
    assert isinstance(output["result"], str)
