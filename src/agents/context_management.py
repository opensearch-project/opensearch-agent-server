"""Context management for the agents (issue #138).

- ``SummarizingConversationManager`` (a Strands ``ConversationManager``) proactively
  summarizes the oldest messages before the context window fills, preserving key
  information while freeing space for new turns.
- ``ContextOffloader`` (a Strands ``Plugin``) intercepts large tool results at execution
  time, stores each block in an in-memory backend, and keeps a short preview in context;
  it registers a ``retrieve_offloaded_content`` tool so the agent can fetch the full
  content on demand.

The ART specialists get both via their ``Agent(...)`` constructor. The default and
ART-orchestrator agents are wrapped by ``ag_ui_strands``, which drops ``plugins`` (and
shares one ``conversation_manager``) on its per-thread rebuild, so
``apply_context_management`` re-applies them per thread (see ``agent_orchestrator``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strands.agent.conversation_manager import (
    ConversationManager,
    SummarizingConversationManager,
)
from strands.vended_plugins.context_offloader import ContextOffloader, InMemoryStorage

from utils.logging_helpers import get_logger, log_debug_event

if TYPE_CHECKING:
    from strands import Agent
    from strands.plugins import Plugin

logger = get_logger(__name__)

# SummarizingConversationManager — compression triggers on a ratio of the model's context
# window, so 0.85 fires near the limit on any size (170k on Sonnet's 200k, ~850k on 1M).
SUMMARY_RATIO = 0.3  # summarize the oldest 30% of messages when compression fires
COMPRESSION_THRESHOLD = 0.85  # fire at 85% of the context window
PRESERVE_RECENT_MESSAGES = 6  # keep the 6 most-recent messages verbatim

# ContextOffloader — offload a tool result over MAX_RESULT_TOKENS, keeping a
# PREVIEW_TOKENS-sized preview in context; full content stays retrievable on demand.
MAX_RESULT_TOKENS = 1_500
PREVIEW_TOKENS = 750

# Sentinel marking the offloader as already applied, so re-application is a no-op.
_APPLIED_ATTR = "_context_offloader_applied"


def create_conversation_manager() -> SummarizingConversationManager:
    """Build a summarizing manager configured for proactive compression."""
    log_debug_event(
        logger,
        f"Context management: summary_ratio={SUMMARY_RATIO}, "
        f"compression_threshold={COMPRESSION_THRESHOLD}, "
        f"preserve_recent_messages={PRESERVE_RECENT_MESSAGES}.",
        "context_management.manager_created",
        summary_ratio=SUMMARY_RATIO,
        compression_threshold=COMPRESSION_THRESHOLD,
        preserve_recent_messages=PRESERVE_RECENT_MESSAGES,
    )
    return SummarizingConversationManager(
        summary_ratio=SUMMARY_RATIO,
        preserve_recent_messages=PRESERVE_RECENT_MESSAGES,
        proactive_compression={"compression_threshold": COMPRESSION_THRESHOLD},
    )


def create_context_offloader() -> ContextOffloader:
    """Build a ``ContextOffloader`` backed by a fresh in-memory store per call."""
    return ContextOffloader(
        storage=InMemoryStorage(),
        max_result_tokens=MAX_RESULT_TOKENS,
        preview_tokens=PREVIEW_TOKENS,
    )


def context_management_plugins() -> list[Plugin]:
    """Return the context-management plugins for ``Agent(plugins=...)`` (a fresh offloader)."""
    return [create_context_offloader()]


def _detach_manager_hooks(agent: Agent, manager: ConversationManager) -> None:
    """Detach ``manager``'s hooks so the inherited one (ag_ui_strands >= 0.1.9 shares it) doesn't fire compression twice."""
    registered = agent.hooks._registered_callbacks
    for event_type, callbacks in registered.items():
        registered[event_type] = [
            cb for cb in callbacks if getattr(cb, "__self__", None) is not manager
        ]


def apply_context_management(agent: Agent) -> None:
    """Attach a fresh manager and offloader onto an already-built ``agent`` in place.

    Used for the per-thread agents ``ag_ui_strands`` rebuilds with constructor kwargs
    dropped; directly-built agents use the constructor instead. Idempotent: the
    ``_APPLIED_ATTR`` guard makes a repeated call a no-op (a second offloader add would raise).
    """
    if getattr(agent, _APPLIED_ATTR, False):
        return

    _detach_manager_hooks(agent, agent.conversation_manager)

    manager = create_conversation_manager()
    agent.conversation_manager = manager
    manager.register_hooks(agent.hooks)  # wire the proactive-compression hook
    # Registers the offloader's hook + retrieve_offloaded_content tool. The sentinel
    # above ensures this runs once per agent (a second add would raise on duplicate).
    agent._plugin_registry.add_and_init(create_context_offloader())

    setattr(agent, _APPLIED_ATTR, True)
