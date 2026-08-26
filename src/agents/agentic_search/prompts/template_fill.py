"""Prompt and cached system blocks for search-template fill (design part 2).

Template fill hands the model a per-template ``FillTemplate`` tool whose fields
ARE the template's Mustache parameters (1:1); the model returns only values
(~5-30 tokens) and OpenSearch renders them into the stored body. The prompt is
deliberately tiny — no DSL rules, no examples, no mapping — because the win here
is output tokens, not caching (§7), and the typed tool schema (per-field
descriptions + enum options) already carries the guidance the model needs.
"""

from __future__ import annotations

from strands.types.content import SystemContentBlock

FILL_MODEL_NAME = "FillTemplate"

FILL_SYSTEM_PROMPT = (
    "You extract search parameters from a user's question to fill a predefined "
    "OpenSearch search template.\n"
    "Call the FillTemplate tool exactly once. Fill only the parameters the "
    "question clearly implies; leave everything else unset — do not guess.\n"
    "Put ONLY content/topic words in any free-text query parameter — never counts, "
    "filters, sort terms, or field names.\n"
    "For enum parameters, choose only from the options that parameter allows.\n"
    "If the question needs something these parameters cannot express — a field not "
    "listed here, prefix/wildcard/fuzzy matching, aggregations, an unsupported range, "
    "or custom scoring — set cannot_express=true and leave the other parameters unset. "
    "Do not force an approximate fill; abstaining routes the question to a more capable "
    "path."
)


def build_fill_system_blocks(system_prompt: str) -> list[SystemContentBlock]:
    """Wrap a system-prompt string in the standard content blocks + cache point.

    Factored out so a prompt-sweep experiment can build blocks from an alternate
    prompt string; the module default :data:`FILL_SYSTEM_BLOCKS` is this applied to
    :data:`FILL_SYSTEM_PROMPT`.
    """
    return [
        {"text": system_prompt},
        {"cachePoint": {"type": "default"}},
    ]


# Sent as content blocks with a trailing cache point, mirroring the direct-DSL
# SYSTEM_BLOCKS. The fill prompt is small, so caching is a minor lever here (a short
# prefix on a fast model may fall under the per-checkpoint token floor); the point
# is harmless and keeps the two paths symmetric.
FILL_SYSTEM_BLOCKS: list[SystemContentBlock] = build_fill_system_blocks(
    FILL_SYSTEM_PROMPT
)

FILL_USER_PROMPT = """\
Question: {question}

Fill the FillTemplate tool's parameters for this question.
"""
