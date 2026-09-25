"""
Prometheus metrics.

Each one exists to answer a question you will actually ask:

  "is it slow, and WHICH STAGE is slow?"      -> rag_stage_duration_seconds
  "is retrieval finding nothing?"             -> rag_zero_result_total
  "am I about to burn my API quota?"          -> airlabs_quota_remaining
  "is the model fabricating citations?"       -> rag_validation_failures_total
  "how much context am I actually sending?"   -> rag_context_tokens
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --------------------------------------------------------------- pipeline
rag_requests_total = Counter(
    "rag_requests_total", "RAG requests handled", ["intent", "outcome"]
)

rag_stage_duration_seconds = Histogram(
    "rag_stage_duration_seconds",
    "Duration of each pipeline stage",
    ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

rag_end_to_end_seconds = Histogram(
    "rag_end_to_end_seconds",
    "Total time from question to validated answer",
    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 40),
)

# -------------------------------------------------------------- retrieval
rag_candidates = Histogram(
    "rag_candidates",
    "Candidates surviving fusion, before rerank",
    buckets=(0, 1, 5, 10, 25, 50, 100),
)

rag_zero_result_total = Counter(
    "rag_zero_result_total", "Queries where retrieval returned nothing"
)

rag_rerank_top_score = Histogram(
    "rag_rerank_top_score",
    "Cross-encoder score of the best chunk; a confidence proxy",
    buckets=(-10, -5, -2, 0, 2, 5, 8, 10),
)

# ------------------------------------------------------------- generation
rag_context_tokens = Histogram(
    "rag_context_tokens",
    "Estimated tokens in the assembled context",
    buckets=(500, 1000, 2000, 3000, 4000, 6000, 8000, 12000),
)

rag_validation_failures_total = Counter(
    "rag_validation_failures_total", "Answer validation failures", ["kind"]
)

rag_retries_total = Counter(
    "rag_retries_total", "Generations retried after failed validation"
)

rag_abstentions_total = Counter(
    "rag_abstentions_total", "Answers that correctly declined to answer"
)


rag_followups_total = Counter(
    "rag_followups_total", "Follow-ups answered without a model call",
    ["template", "material"],   # material: quoted|none
)

# ------------------------------------------------------------------ tools
airlabs_calls_total = Counter(
    "airlabs_calls_total", "Flight lookups", ["source"]  # live|cache|fixture|blocked
)

airlabs_quota_remaining = Gauge(
    "airlabs_quota_remaining", "AirLabs free-tier API calls left this month"
)

# ------------------------------------------------------------------ index
index_chunks_total = Gauge("index_chunks_total", "Chunks currently indexed")

# ------------------------------------------------------------ confidence gate
rag_confidence = Histogram(
    "rag_confidence",
    "Pre-generation confidence score from the retrieval-quality gate",
    buckets=(0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0),
)

rag_gate_abstentions_total = Counter(
    "rag_gate_abstentions_total",
    "Refusals issued BEFORE generation because retrieval was too weak",
    ["reason"],
)

# -------------------------------------------------------------- LLM calls
# Labels are bounded: provider is the LLM_PROVIDER enum; outcome and reason are
# fixed strings set in generation.LLMClient, never exception text.
llm_calls_total = Counter(
    "llm_calls_total", "Generation calls by final outcome",
    ["provider", "outcome"],   # ok|truncated|empty|malformed|request_error|transient_error
)

llm_retries_total = Counter(
    "llm_retries_total", "Generation call retries",
    ["provider", "reason"],    # rate_limited|server_error|timeout|connection
)

# --------------------------------------------------------------------- cost
# Emitted only when the provider reports usage (tokens) and prices are configured
# (cost). An absent series means UNAVAILABLE, not zero.
rag_query_cost_usd = Histogram(
    "rag_query_cost_usd",
    "Estimated generation cost of one LLM call, from reported usage and configured prices",
    buckets=(0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05),
)

rag_tokens_total = Counter(
    "rag_tokens_total", "Tokens reported by the LLM provider", ["kind"]  # prompt|completion
)
