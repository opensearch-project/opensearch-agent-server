"""Centralized input validation using Pydantic validators.

This module provides validated models for request inputs with comprehensive
validation logic and actionable error messages.

**Key Features:**
- Pydantic field validators for type checking and format validation
- Model validators for cross-field validation
- Actionable error messages with guidance for fixing issues
- Centralized validation logic for consistency

**Usage Example:**
```python
from server.validators import ValidatedRunAgentInput

@app.post("/runs")
async def create_run(input_data: ValidatedRunAgentInput) -> StreamingResponse:
    # input_data is guaranteed to be valid at this point
    thread_id = input_data.thread_id
    run_id = input_data.run_id
    # ... rest of handler logic
```
"""

from __future__ import annotations

import json
from typing import Any

from ag_ui.core import Context, Message, RunAgentInput, Tool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel
from pydantic_core import PydanticCustomError

from server.constants import DEFAULT_MAX_TOOL_RESULT_BYTES

# On a continuation run a client-returned tool result is untrusted input that flows straight
# into the model context, so bound it at the API boundary (context management, #138).
_MAX_TOOL_RESULT_BYTES = DEFAULT_MAX_TOOL_RESULT_BYTES

__all__ = ["ValidatedRunAgentInput"]


class ValidatedRunAgentInput(BaseModel):
    """Validated wrapper for RunAgentInput with comprehensive validation.

    This model wraps the external RunAgentInput model and adds validation
    with actionable error messages. All validation is performed using
    Pydantic validators for consistency and better error reporting.

    The model accepts both camelCase (threadId) and snake_case (thread_id)
    field names for compatibility with the AG-UI protocol.

    **Architecture Note: Boundary Pattern**

    This model implements a boundary pattern that separates API concerns from
    internal implementation:

    The ag_ui.core RunAgentInput type requires thread_id and run_id. This model
    enforces both at the API boundary (422 when missing). StrandsAgent.run()
    expects input_data to have both; production callers use to_run_agent_input().

    1. **API Boundary (this model):**
       - Accepts camelCase JSON (matching AG-UI protocol spec: "threadId", "forwardedProps")
       - Provides comprehensive validation with actionable error messages
       - Used in FastAPI route handlers as request body type
       - Serializes to camelCase JSON for responses

    2. **Internal Code (RunAgentInput):**
       - Uses snake_case (Python convention: "thread_id", "forwarded_props")
       - Required by external `ag_ui.core` library (cannot be changed)
       - Used by internal functions: `StrandsAgent.run()`, `handle_state_snapshot()`, etc.

    3. **Conversion:**
       - Conversion happens at the boundary via `to_run_agent_input()`
       - Single, explicit conversion point maintains clear separation
       - Internal code remains Pythonic while API matches protocol spec

    **Why This Pattern?**

    - **Protocol Compliance:** AG-UI protocol specification uses camelCase JSON
    - **Python Conventions:** Internal code follows Python snake_case conventions
    - **External Library Constraint:** `ag_ui.core.RunAgentInput` is external and uses snake_case
    - **Clear Separation:** Validation at boundary, business logic uses library types

    See `PYTHON_CONCEPTS.md` for details on field naming conventions and when to use
    camelCase vs snake_case.

    Attributes:
        thread_id: Non-empty thread identifier string
        run_id: Non-empty run identifier string
        parent_run_id: Optional parent run identifier
        state: Agent state (any type)
        messages: Non-empty list of messages
        tools: List of available tools
        context: List of context objects
        forwarded_props: Forwarded properties (any type)

    Raises:
        pydantic.ValidationError: If any field fails validation with actionable error message
            (FastAPI automatically converts this to HTTP 422 response)

    """

    model_config = ConfigDict(
        extra="allow",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    thread_id: str = Field(
        ...,
        description="Thread identifier for the conversation",
        min_length=1,
    )
    run_id: str = Field(
        ...,
        description="Run identifier for this agent execution",
        min_length=1,
    )
    parent_run_id: str | None = Field(
        None,
        description="Optional parent run identifier",
    )
    state: Any = Field(
        ...,
        description=(
            "Current state of the agent. "
            "Type is intentionally Any to allow flexible state structures."
        ),
    )
    messages: list[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1,
    )
    tools: list[Tool | dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of tools available to the agent. "
            "Accepts Tool objects or dict representations for flexibility."
        ),
    )
    context: list[Context | dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of context objects provided to the agent. "
            "Accepts Context objects or dict representations for flexibility."
        ),
    )
    forwarded_props: Any = Field(
        default=None,
        description=(
            "Additional properties forwarded to the agent. "
            "Type is intentionally Any to allow flexible property structures."
        ),
    )

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, v: str, _info: ValidationInfo) -> str:
        """Validate thread_id is a non-empty string.

        Args:
            v: Thread ID value to validate
            _info: Validation context information (unused; required by Pydantic API)

        Returns:
            Validated thread_id string

        Raises:
            PydanticCustomError: If thread_id is empty or whitespace-only

        """
        if not v or not v.strip():
            raise PydanticCustomError(
                "thread_id_empty",
                (
                    "thread_id must be a non-empty string. "
                    "Please provide a valid thread identifier. "
                    "Example: 'thread-123' or 'conv-abc-xyz'"
                ),
                {"field": "thread_id"},
            )
        return v.strip()

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str, _info: ValidationInfo) -> str:
        """Validate run_id is a non-empty string.

        Args:
            v: Run ID value to validate
            _info: Validation context information (unused; required by Pydantic API)

        Returns:
            Validated run_id string

        Raises:
            PydanticCustomError: If run_id is empty or whitespace-only

        """
        if not v or not v.strip():
            raise PydanticCustomError(
                "run_id_empty",
                (
                    "run_id must be a non-empty string. "
                    "Please provide a valid run identifier. "
                    "Example: 'run-456' or 'exec-789'"
                ),
                {"field": "run_id"},
            )
        return v.strip()

    @field_validator("parent_run_id")
    @classmethod
    def validate_parent_run_id(cls, v: str | None, _info: ValidationInfo) -> str | None:
        """Validate parent_run_id if provided.

        Args:
            v: Parent run ID value to validate (may be None)
            _info: Validation context information (unused; required by Pydantic API)

        Returns:
            Validated parent_run_id string or None

        Raises:
            PydanticCustomError: If parent_run_id is provided but empty

        """
        if v is not None and (not v or not v.strip()):
            raise PydanticCustomError(
                "parent_run_id_empty",
                (
                    "parent_run_id must be a non-empty string if provided. "
                    "Either omit this field or provide a valid parent run identifier."
                ),
                {"field": "parent_run_id"},
            )
        return v.strip() if v else None

    @field_validator("messages")
    @classmethod
    def validate_messages(
        cls, v: list[Message], _info: ValidationInfo
    ) -> list[Message]:
        """Validate messages list is non-empty and contains valid messages.

        Args:
            v: Messages list to validate
            _info: Validation context information (unused; required by Pydantic API)

        Returns:
            Validated messages list

        Raises:
            PydanticCustomError: If messages list is empty or contains invalid messages

        """
        if not v:
            raise PydanticCustomError(
                "messages_empty",
                (
                    "messages must be a non-empty array. "
                    "Please provide at least one message in the conversation. "
                    "Example: [{'role': 'user', 'content': 'Hello'}]"
                ),
                {"field": "messages"},
            )

        # Validate each message has required fields
        for idx, msg in enumerate(v):
            if not hasattr(msg, "role") or not msg.role:
                raise PydanticCustomError(
                    "message_missing_role",
                    (
                        f"messages[{idx}].role is required and must be a non-empty string. "
                        f"Valid roles are: 'user', 'assistant', 'system', 'developer', 'tool', 'activity'. "
                        f"Please ensure each message has a valid role field."
                    ),
                    {"field": f"messages[{idx}].role", "index": idx},
                )

            # Content validation: required unless assistant message with tool_calls
            has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
            is_assistant_with_tools = (
                getattr(msg, "role", None) == "assistant" and has_tool_calls
            )

            if not is_assistant_with_tools:
                # Content is required for non-assistant messages or assistant without tool calls
                content = getattr(msg, "content", None)
                if content is None:
                    raise PydanticCustomError(
                        "message_missing_content",
                        (
                            f"messages[{idx}].content is required. "
                            f"Please provide content for this message. "
                            f"Example: {{'role': '{getattr(msg, 'role', 'user')}', 'content': 'Your message here'}}"
                        ),
                        {"field": f"messages[{idx}].content", "index": idx},
                    )

            # Bound a client-returned tool result at the API boundary (see _MAX_TOOL_RESULT_BYTES).
            if getattr(msg, "role", None) == "tool":
                content = getattr(msg, "content", None)
                try:
                    size = (
                        len(content.encode("utf-8"))
                        if isinstance(content, str)
                        else len(json.dumps(content, default=str).encode("utf-8"))
                    )
                except (TypeError, ValueError):
                    size = _MAX_TOOL_RESULT_BYTES + 1
                if size > _MAX_TOOL_RESULT_BYTES:
                    raise PydanticCustomError(
                        "tool_result_too_large",
                        "messages[{index}] tool result is too large ({size} bytes, max {max} bytes).",
                        {
                            "field": f"messages[{idx}]",
                            "index": idx,
                            "size": size,
                            "max": _MAX_TOOL_RESULT_BYTES,
                        },
                    )

        return v

    @model_validator(mode="after")
    def validate_model(self) -> ValidatedRunAgentInput:
        """Perform cross-field validation after all fields are validated.

        This validator runs after individual field validators and can
        perform validation that depends on multiple fields.

        Returns:
            Self (validated model instance)

        Raises:
            PydanticCustomError: If cross-field validation fails

        """
        # Additional cross-field validations can be added here
        # For example, checking that thread_id and run_id are different
        if self.thread_id == self.run_id:
            raise PydanticCustomError(
                "thread_run_id_same",
                (
                    "thread_id and run_id must be different. "
                    "Please use distinct identifiers for thread and run."
                ),
                {"thread_id": self.thread_id, "run_id": self.run_id},
            )

        return self

    def to_run_agent_input(self) -> RunAgentInput:
        """Convert validated model to RunAgentInput instance.

        Returns:
            RunAgentInput instance with validated data

        """
        return RunAgentInput(
            thread_id=self.thread_id,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            state=self.state,
            messages=self.messages,
            tools=self.tools,
            context=self.context,
            forwarded_props=self.forwarded_props,
        )

    @classmethod
    def from_run_agent_input(cls, input_data: RunAgentInput) -> ValidatedRunAgentInput:
        """Create ValidatedRunAgentInput from RunAgentInput instance.

        This method validates an existing RunAgentInput instance and
        returns a validated wrapper.

        Args:
            input_data: RunAgentInput instance to validate

        Returns:
            ValidatedRunAgentInput instance

        Raises:
            pydantic.ValidationError: If validation fails
                (FastAPI automatically converts this to HTTP 422 response)

        """
        return cls(
            thread_id=input_data.thread_id,
            run_id=input_data.run_id,
            parent_run_id=getattr(input_data, "parent_run_id", None),
            state=input_data.state,
            messages=input_data.messages,
            tools=input_data.tools,
            context=input_data.context,
            forwarded_props=input_data.forwarded_props,
        )
