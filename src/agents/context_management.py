"""Context management for the agents (issue #138).

Maps the ml-commons AG-UI chat agent's ``context_management`` onto Strands: proactive
summarization (``SummarizingConversationManager``) plus per-tool-result truncation
(``ToolOutputTruncationHook``), so long conversations and large tool outputs don't
overflow the context window.

The ART specialists get this from their ``Agent(...)`` constructor. The default and
ART-orchestrator agents are wrapped by ``ag_ui_strands.StrandsAgent`` (0.1.1), which
rebuilds a per-thread ``Agent`` forwarding only ``model``/``system_prompt``/``tools``
and dropping ``conversation_manager`` + ``hooks``; ``apply_context_management`` re-applies
both per thread (see ``agent_orchestrator``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strands.agent.conversation_manager import SummarizingConversationManager
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterToolCallEvent

from utils.logging_helpers import get_logger, log_debug_event

if TYPE_CHECKING:
    from strands import Agent

logger = get_logger(__name__)

# Strands' proactive compression triggers on a RATIO of the model's window, not an absolute
# count. 0.85 * 200_000 (Sonnet's window) = 170_000, matching ml-commons' tokens_exceed:
# 170000, and fires near the limit on any window (e.g. ~850k on 1M). Matches the reference agent.
SUMMARIZATION_TRIGGER_TOKENS = 170_000
CONTEXT_WINDOW_LIMIT = 200_000
SUMMARIZATION_COMPRESSION_THRESHOLD = (
    SUMMARIZATION_TRIGGER_TOKENS / CONTEXT_WINDOW_LIMIT
)  # 0.85
PRESERVE_RECENT_MESSAGES = 6
MAX_TOOL_OUTPUT_CHARS = 100_000


class ToolOutputTruncationHook(HookProvider):
    """Truncates each tool result text block to ``max_chars`` after the tool runs."""

    def __init__(self, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> None:
        self._max_chars = max_chars

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AfterToolCallEvent, self._truncate)

    def _truncate(self, event: AfterToolCallEvent) -> None:
        result = getattr(event, "result", None)
        # ToolResult is a dict on success; skip anything else (e.g. an Exception).
        if not isinstance(result, dict):
            return
        content = result.get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str) and len(text) > self._max_chars:
                block["text"] = (
                    text[: self._max_chars]
                    + f"\n\n[output truncated to {self._max_chars} characters]"
                )


def create_conversation_manager() -> SummarizingConversationManager:
    """Build a manager that proactively summarizes the oldest messages near the limit.

    Uses the fixed ``SUMMARIZATION_COMPRESSION_THRESHOLD`` ratio, matching the reference agent.

    Returns:
        A ``SummarizingConversationManager`` configured for proactive compression.
    """
    log_debug_event(
        logger,
        f"Context management: summarization trigger={SUMMARIZATION_TRIGGER_TOKENS} "
        f"tokens, compression_threshold={SUMMARIZATION_COMPRESSION_THRESHOLD:.3f}, "
        f"preserve_recent_messages={PRESERVE_RECENT_MESSAGES}.",
        "context_management.manager_created",
        summarization_trigger_tokens=SUMMARIZATION_TRIGGER_TOKENS,
        compression_threshold=SUMMARIZATION_COMPRESSION_THRESHOLD,
        preserve_recent_messages=PRESERVE_RECENT_MESSAGES,
    )
    return SummarizingConversationManager(
        preserve_recent_messages=PRESERVE_RECENT_MESSAGES,
        proactive_compression={
            "compression_threshold": SUMMARIZATION_COMPRESSION_THRESHOLD
        },
    )


def apply_context_management(agent: Agent) -> None:
    """Attach context management onto an already-built ``agent`` in place.

    For the per-thread agents ``ag_ui_strands`` rebuilds with constructor kwargs dropped;
    directly-built agents should use the constructor. A fresh manager is created per call
    since it is stateful (holds the running summary) — sharing would leak across sessions.

    Args:
        agent: An already-constructed Strands Agent to augment in place.
    """
    manager = create_conversation_manager()
    agent.conversation_manager = manager
    # Wires the manager's proactive-compression BeforeModelCallEvent hook.
    manager.register_hooks(agent.hooks)
    ToolOutputTruncationHook().register_hooks(agent.hooks)
