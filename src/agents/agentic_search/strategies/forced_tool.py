"""Forced tool-call generation over Bedrock ``converse_stream``.

Registers a Pydantic model as a tool and forces ``toolChoice`` so the model emits
the tool call immediately, with no leading free text. The high-level strands
``Agent`` API does not expose ``toolChoice``, so the call is driven directly; this
also means the SDK's automatic retry-on-non-tool-response is not available, so an
empty or malformed tool call raises ``ValueError`` for the caller to handle.

This is a workaround. It is Bedrock-specific — it uses the botocore
``converse_stream`` client directly — so callers must gate it with
:func:`supports_forced_tool` and use the portable strands ``structured_output``
path for other providers (e.g. Ollama). It can be removed once strands exposes
``tool_choice`` on ``structured_output``.

Tracking: https://github.com/opensearch-project/opensearch-agent-server/issues/149
Upstream: https://github.com/strands-agents/harness-sdk/issues/3336
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError
from strands.tools.structured_output import convert_pydantic_to_tool_spec

logger = logging.getLogger(__name__)


def supports_forced_tool(model: Any) -> bool:
    """Whether ``model`` supports the forced-tool path (Bedrock ``converse_stream``).

    Only Bedrock models expose the botocore ``converse_stream`` client this path
    drives directly. Other providers (e.g. Ollama) must use the portable strands
    ``structured_output`` path instead.
    """
    client = getattr(model, "client", None)
    return client is not None and hasattr(client, "converse_stream")


def forced_tool_fill(
    *,
    model: Any,
    schema_model: type[BaseModel],
    system_blocks: list[dict],
    user_message: str,
    tool_spec: dict[str, Any] | None = None,
) -> BaseModel:
    """Force a single tool call for ``schema_model`` and return the validated instance.

    Args:
        model: The shared strands ``BedrockModel``; its pooled botocore client
            (``model.client``) and ``model.config`` (``model_id``, optional
            ``temperature``) drive the call directly.
        schema_model: The Pydantic model registered as the forced tool. Its
            validated instance is returned.
        system_blocks: Bedrock system content blocks (may carry a cache point);
            copied defensively before sending.
        user_message: The user turn's text.
        tool_spec: Optional precomputed ``convert_pydantic_to_tool_spec`` result.
            When ``None`` it is computed from ``schema_model`` here.

    Returns:
        The validated ``schema_model`` instance.

    Raises:
        ValueError: The forced tool produced no input, or the input failed
            validation against ``schema_model``.
    """
    spec = tool_spec if tool_spec is not None else convert_pydantic_to_tool_spec(schema_model)
    tool_name = spec["name"]

    client = model.client  # botocore bedrock-runtime client (pooled connection)
    converse_kwargs: dict[str, Any] = {
        "modelId": model.config.get("model_id"),
        "system": [dict(b) for b in system_blocks],
        "messages": [{"role": "user", "content": [{"text": user_message}]}],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool_name,
                        "description": spec.get("description", "Emit the result."),
                        "inputSchema": spec["inputSchema"],
                    }
                }
            ],
            "toolChoice": {"tool": {"name": tool_name}},
        },
    }
    # Driving converse_stream directly bypasses the strands request builder, so
    # apply the model's configured temperature here if one is set.
    temperature = model.config.get("temperature")
    if temperature is not None:
        converse_kwargs["inferenceConfig"] = {"temperature": temperature}

    resp = client.converse_stream(**converse_kwargs)

    # Accumulate the forced tool's input JSON from the stream.
    tool_input = ""
    for event in resp["stream"]:
        tool_use = event.get("contentBlockDelta", {}).get("delta", {}).get("toolUse")
        if tool_use and "input" in tool_use:
            tool_input += tool_use["input"]

    if not tool_input.strip():
        raise ValueError("forced tool call produced no input")
    try:
        return schema_model.model_validate_json(tool_input)
    except ValidationError as e:
        raise ValueError(f"forced tool input failed validation: {e}") from e
