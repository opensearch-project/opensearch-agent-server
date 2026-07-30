# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""Optional ``/invoke`` response envelopes, selected by the ``response_format`` field.

These shape *any* agent's string reply for a specific consumer; they are not tied
to a particular agent. Today the only envelope is ml-commons' ``inference_results``
passthrough shape, used by the agentic-search connector.
"""

from __future__ import annotations

from server.types import InferenceResultsResponse


def wrap_inference_results(result: str) -> InferenceResultsResponse:
    """Wrap an agent's string reply in the ml-commons inference_results envelope.

    A connector's built-in ``mlcommons.passthrough`` post-processor copies
    ``output[0].result`` into ``ModelTensor.result`` (the field neural-search reads).
    """
    return {
        "inference_results": [
            {
                "output": [{"name": "response", "result": result}],
                "status_code": 200,
            }
        ]
    }
