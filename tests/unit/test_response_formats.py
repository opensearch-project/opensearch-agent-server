"""Unit tests for the /invoke response envelopes."""

from __future__ import annotations

import pytest

from server.response_formats import wrap_inference_results

pytestmark = pytest.mark.unit


def test_wrap_puts_result_string_at_output_result():
    dsl = '{"query":{"match_all":{}}}'
    env = wrap_inference_results(dsl)

    output = env["inference_results"][0]["output"][0]
    # Must be a STRING at output[0].result (what the connector passthrough reads).
    assert output["result"] == dsl
    assert isinstance(output["result"], str)
    assert output["name"] == "response"
    assert env["inference_results"][0]["status_code"] == 200


def test_wrap_is_verbatim_not_reparsed():
    # The envelope carries the reply as-is; it does not parse or reshape it.
    assert (
        wrap_inference_results("anything")["inference_results"][0]["output"][0][
            "result"
        ]
        == "anything"
    )
