"""Optional ``/invoke`` response envelopes (``response_format``).

Kept separate from any agent: these shape *any* agent's string reply for a
specific consumer, selected by the request's ``response_format``. Today the only
envelope is ml-commons' ``inference_results`` passthrough shape, used by the
agentic-search connector.
"""

from __future__ import annotations

from typing import Any, TypedDict


class InferenceResultsResponse(TypedDict):
    """ml-commons ``inference_results`` envelope returned by ``/invoke``.

    Used when a caller sets ``response_format=inference_results`` (e.g. the
    agentic-search connector). A connector's built-in passthrough post-processor
    copies ``output[0].result`` into ``ModelTensor.result``, which is the one
    field neural-search reads.
    """

    inference_results: list[dict[str, Any]]


def wrap_inference_results(result: str) -> InferenceResultsResponse:
    """Wrap an agent's string reply in the ml-commons inference_results envelope."""
    return {
        "inference_results": [
            {
                "output": [{"name": "response", "result": result}],
                "status_code": 200,
            }
        ]
    }
