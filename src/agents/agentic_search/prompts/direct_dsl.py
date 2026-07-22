"""Prompt, structured-output schema, and cached system blocks for direct-DSL generation.

The system prompt is a large, static rules-and-examples prefix. It is sent as
Bedrock content blocks with a trailing cache point so it is served from cache on
warm calls (Bedrock only; harmlessly ignored elsewhere); the per-request mapping,
sample document, and question go in the user message, after the cached prefix.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from strands.types.content import SystemContentBlock

# Broad match_all body used as the safe fallback when generation fails or no
# mapping field is relevant to the question.
FALLBACK_DSL = '{"size":10,"query":{"match_all":{}}}'


class EmitSearch(BaseModel):
    """Structured output the model returns: the DSL plus a one-line rationale.

    ``dsl`` is an open object (no rigid sub-schema) so the model can emit any
    valid OpenSearch query body.
    """

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        description="One short line mapping each clause of the DSL to the user's words."
    )
    dsl: dict[str, Any] = Field(
        description="The OpenSearch _search request body as a JSON object."
    )


QUERY_TYPE_RULES = (
    "Use only fields present in the provided mapping; never invent names.\n"
    "Choose query types based on user intent and field types:\n"
    "- match: single-token full-text on analyzed text fields.\n"
    "- match_phrase: multi-token phrases on analyzed text fields (search string contains spaces, hyphens, commas, etc.).\n"
    "- multi_match: when multiple analyzed text fields are equally relevant.\n"
    "- term / terms: exact match on keyword, numeric, boolean.\n"
    "- range: numeric/date comparisons (gt, lt, gte, lte).\n"
    "- bool with must, should, must_not, filter: AND/OR/NOT logic.\n"
    '- wildcard / prefix on keyword: "starts with" / pattern matching.\n'
    "- exists: field presence/absence.\n"
    '- nested query / nested agg: ONLY if the mapping for that exact path (or a parent) has "type":"nested".\n'
    "- neural: semantic similarity on a 'semantic' or 'knn_vector' field (dense). "
    'Use "query_text" and "k"; include "model_id" unless bound in mapping.\n'
    "- neural (top-level): allowed when it's the only relevance clause needed; "
    "otherwise wrap in a bool when combining with filters/other queries.\n"
    "\n"
    "Mechanics:\n"
    "- Put exact constraints (term, terms, range, exists, prefix, wildcard) in bool.filter (non-scoring). "
    "Put full-text relevance (match, match_phrase, multi_match) in bool.must.\n"
    '- Top N items/products/documents: return top hits (set "size": N as an integer) and sort by the relevant metric(s). '
    "Do not use aggregations for item lists.\n"
    '- Neural retrieval size: set "k" >= "size" (e.g. heuristic, k = max(size*5, 100) and k<=ef_search).\n'
    '- Spelling tolerance: match_phrase does NOT support fuzziness; use match or multi_match with "fuzziness": "AUTO" '
    "when tolerant matching is needed.\n"
    "- Text operators (OR vs AND): default to OR for natural-language queries; to tighten, use minimum_should_match "
    '(e.g., "75%" requires ~75% of terms). Use AND only when every token is essential; if order/adjacency matters, '
    "use match_phrase. (Applies to match/multi_match.)\n"
    '- Numeric note: use ONLY integers for size and k (e.g., "size": 5), not floats (wrong e.g., "size": 5.0).\n'
)

AGGREGATION_RULES = (
    "Aggregations (counts, averages, grouped summaries, distributions):\n"
    "- Use aggregations when the user asks for grouped summaries (e.g., counts by category, averages by brand, "
    "or top N categories/brands).\n"
    "- terms on field.keyword or numeric for grouping / top N groups (not items).\n"
    "- Metric aggs (avg, min, max, sum, stats, cardinality) on numeric fields.\n"
    "- date_histogram, histogram, range for distributions.\n"
    '- Always set "size": 0 when only aggregations are needed.\n'
    '- Use sub-aggregations + order for "top N groups by metric".\n'
    "- If grouping/filtering exactly on a text field, use its .keyword sub-field when present.\n"
)

SEMANTIC_SEARCH_RULES = (
    "NEURAL / SEMANTIC SEARCH\n"
    "When to use:\n"
    '- The intent is conceptual/semantic ("about", "similar to", long phrases, synonyms, multilingual, ambiguous), '
    "and the mapping has:\n"
    '  - type: "semantic", or\n'
    '  - type: "knn_vector".\n'
    "- You also have exact filters (term/range/etc.) but text relevance still matters -> add neural on that text field.\n"
    "- The user explicitly asks for semantic/neural/vector/embedding search.\n"
    "When NOT to use:\n"
    "- The request is purely structured/exact (IDs, codes, only term/range).\n"
    '- No suitable "semantic" or "knn_vector" field exists.\n'
    "- No Model ID found for neural search.\n"
    "How to query:\n"
    '- Use the "neural" clause against the chosen field.\n'
    '- Required: "query_text" and "k".\n'
    '- "model_id" rules:\n'
    '  - For "semantic" fields, model usually bound in mapping -> omit unless overriding.\n'
    '  - For "knn_vector", include "model_id" unless a default is bound elsewhere.\n'
    "  - If model ID is not found, do not generate query with Neural clause.\n"
    "Top-level usage:\n"
    '- If there are no filters/other clauses, "neural" MAY be the root query (no bool).\n'
    "- Use a bool wrapper only when combining with filters or additional queries; keep exact filters in bool.filter.\n"
    "Sizing:\n"
    '- "size": N is the returned hits.\n'
    '- Set "k" >= "size" (heuristic: k (int) = max(size*5, 100), reasonable cap ~ 1000).\n'
    "Field choice:\n"
    "- Prefer a field that semantically represents intent (e.g., description/title/content).\n"
    "- If multiple candidates exist, pick the single best; add more only if clearly beneficial.\n"
    "Fallback:\n"
    "- If no suitable neural field exists or if no model id is found, do NOT add a neural clause; "
    "proceed with classic DSL or fall back to the default query if nothing relevant exists.\n"
)

DATE_RULES = (
    "DATE RULES\n"
    "- Use range on date/date_nanos in bool.filter.\n"
    "- Emit ISO 8601 UTC ('Z') bounds; don't set time_zone for explicit UTC. (now is UTC)\n"
    "- Date math: now+-N{y|M|w|d|h|m|s} (M=month, m=minute; e.g., now-7d .. now = last 7 days).\n"
    '- Rounding: "/UNIT" floors to start (now/d, now/w, now/M, now/y). '
    "Examples: last full day -> now-1d/d .. now/d; last full month -> now-1M/M .. now/M.\n"
    "- End boundaries: prefer the next unit's start (avoid 23:59:59).\n"
    '- Formats: only add "format" when inputs aren\'t default; epoch_millis allowed.\n'
    "- Buckets: use date_histogram (set calendar_interval or fixed_interval); "
    "add time_zone only when local day/week/month buckets are required.\n"
)

FIELD_SELECTION_AND_PROXYING = (
    "Goal: pick the smallest set of mapping fields that best capture the user's intent.\n"
    "Query Fields: when provided, and present in the mapping, prioritize using them; "
    "ignore any that are not in the mapping.\n"
    "Proxy Rule (mandatory): If at least one field is even loosely related to the intent, you MUST proceed using "
    "the best available proxy fields. Do NOT fall back to the default query due to ambiguity.\n"
    "Selection steps:\n"
    "- Harvest candidates from the question (entities, attributes, constraints).\n"
    "- From query_fields (that exist) and the index mapping, choose fields that map to those candidates and the "
    "user intent, even if only loosely (use reasonable proxies).\n"
    "- Ignore other fields that don't help answer the question.\n"
    "- Micro Self-Check (silent): verify chosen fields exist; if any don't, swap to the closest mapped proxy and "
    "continue. Only if no remotely relevant fields exist at all, use the default query.\n"
)

PROMPT_PREFIX = (
    "==== PURPOSE ====\n"
    "You are an OpenSearch DSL expert. Convert a natural-language question into a strict JSON OpenSearch query body.\n\n"
    "==== RULES ====\n"
    + QUERY_TYPE_RULES
    + "\n"
    + AGGREGATION_RULES
    + "\n"
    + DATE_RULES
    + "\n"
    + SEMANTIC_SEARCH_RULES
    + "\n"
    + "==== FIELD SELECTION & PROXYING ====\n"
    + FIELD_SELECTION_AND_PROXYING
)

OUTPUT_FORMAT_INSTRUCTIONS = (
    "==== OUTPUT FORMAT ====\n"
    "- Put the OpenSearch request body (a single JSON object) in the `dsl` field of the EmitSearch tool call.\n"
    "- Use valid JSON only: standard double quotes for all keys/strings; no comments; no trailing commas.\n"
    "- In the `reason` field, briefly map each clause to the user's words.\n"
    "- If the request truly cannot be fulfilled because no remotely relevant fields exist, set `dsl` to EXACTLY:\n"
    + FALLBACK_DSL
    + "\n"
)

EXAMPLE_1 = (
    "Example 1 - numeric + date range (merged)\n"
    "Input: Show all products that cost more than 50 dollars in the last 30 days.\n"
    'Mapping: { "properties": { "price": { "type": "float" }, "created_at": { "type": "date" }, "color": { "type": "keyword" } } }\n'
    "Query Fields: [price, created_at]\n"
    "Field selection: relevant=[price(float), created_at(date)]; ignored=[color]\n"
    'Output: { "query": { "bool": { "filter": [{ "range": { "price": { "gt": 50 } } }, { "range": { "created_at": { "gte": "now-30d/d", "lte": "now" } } } ] } } }\n'
)

EXAMPLE_2 = (
    "Example 2 - text match + exact filter (spelling tolerant)\n"
    "Input: Find employees in London who are active.\n"
    'Mapping: { "properties": { "city": { "type": "text", "fields": { "keyword": { "type": "keyword" } } }, "status": { "type": "keyword" }, "notes": { "type": "text" } } }\n'
    "Query Fields: [city, status]\n"
    "Field selection: relevant=[city(text), status(keyword)]; ignored=[notes]\n"
    'Output: { "query": { "bool": { "must": [ { "match": { "city": { "query": "London", "fuzziness": "AUTO" } } } ], "filter": [ { "term": { "status": "active" } } ] } } }\n'
)

EXAMPLE_3 = (
    "Example 3 - match_phrase for multi-token\n"
    "Input: Find employees located in New York City.\n"
    'Mapping: { "properties": { "city": { "type": "text", "fields": { "keyword": { "type": "keyword" } } }, "department": { "type": "keyword" } } }\n'
    'Output: { "query": { "match_phrase": { "city": "New York City" } } }\n'
)

EXAMPLE_4 = (
    "Example 4 - multi_match across fields + SHOULD filters\n"
    'Input: Find profiles mentioning "data engineering" in the title or summary that are research papers or blogs.\n'
    'Mapping: { "properties": { "title": { "type": "text" }, "summary": { "type": "text" }, "type": { "type": "keyword" } } }\n'
    'Output: { "query": { "bool": { "must": [ { "multi_match": { "query": "data engineering", "fields": ["title", "summary"], "fuzziness": "AUTO" } } ], "should": [ { "term": { "type": "research paper" } }, { "term": { "type": "blog" } } ], "minimum_should_match": 1 } } }\n'
)

EXAMPLE_5 = (
    "Example 5 - wildcard + exists (exact filters in bool.filter)\n"
    'Input: Find users whose email starts with "sam" and who have a phone number on file.\n'
    'Mapping: { "properties": { "email": { "type": "keyword" }, "phone": { "type": "keyword" }, "avatar_url": { "type": "keyword" } } }\n'
    "Field selection: relevant=[email(prefix), phone(exists)]; ignored=[avatar_url]\n"
    'Output: { "query": { "bool": { "filter": [ { "prefix": { "email": "sam" } }, { "exists": { "field": "phone" } } ] } } }\n'
)

EXAMPLE_6 = (
    "Example 6 - nested query (only when mapping says nested)\n"
    "Input: Find books where an author's first_name is John AND last_name is Doe.\n"
    'Mapping: { "properties": { "author": { "type": "nested", "properties": { "first_name": { "type": "text", "fields": { "keyword": { "type": "keyword" } } }, "last_name": { "type": "text", "fields": { "keyword": { "type": "keyword" } } } } }, "title": { "type": "text" } } }\n'
    'Output: { "query": { "nested": { "path": "author", "query": { "bool": { "must": [ { "term": { "author.first_name.keyword": "John" } }, { "term": { "author.last_name.keyword": "Doe" } } ] } } } } }\n'
)

EXAMPLE_7 = (
    "Example 7 - terms aggregation\n"
    "Input: Show the number of orders per status.\n"
    'Mapping: { "properties": { "status": { "type": "keyword" }, "order_id": { "type": "keyword" } } }\n'
    'Output: { "size": 0, "aggs": { "orders_by_status": { "terms": { "field": "status" } } } }\n'
)

EXAMPLE_8 = (
    "Example 8 - top N items by metric (hits + sort, no aggs)\n"
    "Input: Show the 5 highest-rated electronics products.\n"
    'Mapping: { "properties": { "category": { "type": "keyword" }, "rating": { "type": "float" }, "reviews_count": { "type": "integer" }, "product_name": { "type": "text" }, "description": { "type": "text" } } }\n'
    "Field selection: relevant=[category(keyword), rating(float), reviews_count(integer), product_name(text), description(text)]\n"
    'Output: { "size": 5, "query": { "bool": { "filter": [ { "term": { "category": "electronics" } } ] } }, "sort": [ { "rating": { "order": "desc" } }, { "reviews_count": { "order": "desc" } } ] }\n'
)

EXAMPLE_9 = (
    "Example 9 - top N categories (grouping via aggs; not for item lists)\n"
    "Input: List the top 3 categories by total sales volume.\n"
    'Mapping: { "properties": { "category": { "type": "text", "fields": { "keyword": { "type": "keyword" } } }, "sales": { "type": "float" }, "region": { "type": "keyword" } } }\n'
    "Field selection: relevant=[category.keyword, sales]; ignored=[region]\n"
    'Output: { "size": 0, "aggs": { "top_categories": { "terms": { "field": "category.keyword", "size": 3, "order": { "total_sales": "desc" } }, "aggs": { "total_sales": { "sum": { "field": "sales" } } } } } }\n'
)

EXAMPLE_10 = (
    "Example 10 - ambiguous mapping, proxy success\n"
    "Input: Give medicines shipped from Vietnam.\n"
    'Mapping: { "properties": { "item_name": { "type": "text" }, "product_category": { "type": "keyword" }, "country": { "type": "keyword" }, "ship_status": { "type": "keyword" }, "notes": { "type": "text" } } }\n'
    "Query Fields: [product_category, origin_country]\n"
    "Field selection: relevant=[product_category, country(proxy for origin), ship_status(proxy for shipped)]; ignored=[notes, item_name]\n"
    'Output: { "query": { "bool": { "filter": [ { "term": { "product_category": "medicines" } }, { "term": { "country": "Vietnam" } }, { "term": { "ship_status": "shipped" } } ] } } }\n'
)

EXAMPLE_11 = (
    "Example 11 - true fallback (no remotely relevant fields)\n"
    "Input: List satellites with periapsis above 400km.\n"
    'Mapping: { "properties": { "name": { "type": "text" }, "color": { "type": "keyword" } } }\n'
    "Output: " + FALLBACK_DSL + "\n"
)

EXAMPLE_12 = (
    "Example 12 - neural preferred with safe fallback (merged)\n"
    'Input: Find articles about "LLM hallucinations". Model Id may or may not be provided.\n'
    'Mapping: { "properties": { "content": {"type":"text"}, "content_vector": {"type":"knn_vector","dimension":768}, "tags": {"type":"keyword"}, "published_at": {"type":"date"} } }\n'
    'Output (with model_id): { "size": 10, "query": { "neural": { "content_vector": { "query_text": "LLM hallucinations", "model_id": "m-dense-001", "k": 200 } } } }\n'
    'Output (fallback without model_id): { "size": 10, "query": { "match": { "content": { "query": "LLM hallucinations" } } } }\n'
)

EXAMPLE_13 = (
    "Example 13 - neural on semantic field + exact filters (mapping includes non-semantic fields)\n"
    'Input: Find "wireless noise cancelling headphones with multipoint" under $200; brand Sony.\n'
    'Mapping: { "properties": { "price": {"type":"float"}, "brand": {"type":"keyword"}, "title": {"type":"text"}, "description": {"type":"semantic", "model_id":"m-sem-123"} } }\n'
    'Output: { "size": 10, "query": { "bool": { "must": [ { "neural": { "description": { "query_text": "wireless noise cancelling headphones with multipoint", "k": 120 } } } ], "filter": [ { "range": { "price": { "lte": 200 } } }, { "term": { "brand": "Sony" } } ] } } }\n'
)

EXAMPLES = (
    "==== EXAMPLES ====\n"
    + EXAMPLE_1
    + EXAMPLE_2
    + EXAMPLE_3
    + EXAMPLE_4
    + EXAMPLE_5
    + EXAMPLE_6
    + EXAMPLE_7
    + EXAMPLE_8
    + EXAMPLE_9
    + EXAMPLE_10
    + EXAMPLE_11
    + EXAMPLE_12
    + EXAMPLE_13
)

SYSTEM_PROMPT = PROMPT_PREFIX + "\n\n" + OUTPUT_FORMAT_INSTRUCTIONS + "\n" + EXAMPLES

# Sent as content blocks with a trailing cache point so Bedrock caches the static
# prefix (~5-minute TTL) and only the per-request tail (mapping + sample + question,
# in the user message) is billed on warm calls. The cache point is a no-op off Bedrock.
SYSTEM_BLOCKS: list[SystemContentBlock] = [
    {"text": SYSTEM_PROMPT},
    {"cachePoint": {"type": "default"}},
]

# The mapping gives field types; the sample document adds one real indexed doc so
# the model can see actual field values (e.g. exact keyword/enum values), not just types.
USER_PROMPT = """\
Question: {question}
Index: {index_name}
Mapping: {mapping}
Sample document from index: {sample_document}

Emit the OpenSearch _search body for this question.
"""
