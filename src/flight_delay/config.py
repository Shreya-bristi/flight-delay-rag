"""
Application configuration loaded from environment variables.

pydantic-settings handles type conversion and validation at startup.
The same settings model is used locally, in Docker, and in Kubernetes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---------------------------------------------------------------- storage
    pg_dsn: str = "postgresql://fdr:fdr@localhost:5432/fdr"

    # --------------------------------------------------------------- embedding
    embedder: Literal["hash", "st"] = "st"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = Field(1024, ge=1)
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    # Device for query embeddings 
    embedding_query_device: str = ""

    # --------------------------------------------------------------- reranking
    reranker: Literal["none", "ce"] = "ce"
    # Serving reranker selected to fit the available deployment hardware
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_max_length: int = Field(512, ge=128, le=8192)
    fuse_rerank_with_dense: bool = True

    # --------------------------------------------------------------- retrieval
    chunk_target_tokens: int = Field(512, ge=32, le=4096)
    chunk_overlap_pct: int = Field(15, ge=0, le=50)
    chunk_breadcrumb: bool = True
    # candidate pool sizes before final context selection
    candidates_k: int = Field(20, ge=1, le=1000)
    sparse_candidates_k: int = Field(5, ge=0, le=1000)
    final_k: int = Field(8, ge=1, le=50)
    # pgvector HNSW search
    hnsw_ef_search: int = Field(100, ge=1, le=1000)
    rrf_k: int = Field(60, ge=1)
    use_dense: bool = True
    use_sparse: bool = True

    # ------------------------------------------------------- source balancing
    # Reserve ontext space for both government and airline evidence
    balance_sources: bool = True
    # Extra candidates from governing-regime regulations
    governing_candidates_k: int = Field(8, ge=0, le=100)
    min_government_sources: int = Field(2, ge=0)
    min_airline_sources: int = Field(2, ge=0)

    # ------------------------------------------- query and candidate shaping
    #  enrich retrieval queries with known route, carrier, and disruption context.
    structured_query: bool = True
    #  remove evidence for disruption types unrelated to the question
    topic_filter: bool = True
    # Maximum chunks from one section
    max_chunks_per_section: int = Field(0, ge=0, le=50)
    # Additional retrieval lanes for specific legal/remedy evidence
    lane_candidates_k: int = Field(4, ge=0, le=50)
    # Retrieve scope evidence for every jurisdiction in scope.
    scope_lanes: bool = True
    # Retrieve US procedural provisions when relevant
    procedural_lanes: bool = True
    # Reserve final-context seats for high-priority lane results
    lane_seats: bool = True
    seat_remedy_lanes: bool = True
    # Retrieve EU/UK cancellation-trigger provisions when relevant
    trigger_lanes: bool = True
    
    lane_seat_section_diverse: bool = False
    max_seats_per_lane: int = Field(1, ge=1, le=3)
    max_lane_seats: int = Field(5, ge=0, le=8)

    min_airline_sources_legal: int = Field(1, ge=0)

    # ------------------------------------------------------------- follow-ups
    # Answer recognized follow-ups from the previous cited answer when possible
    followup_templates: bool = True
    #  Choose whether unmatched follow-ups use a template or the full pipeline
    followup_fallback: Literal["template", "generate"] = "template"

    # -------------------------------------------------------- confidence gate
    confidence_gate_enabled: bool = True
    min_rerank_score: float = -2.0
    min_score_margin: float = Field(0.5, ge=0.0)
    min_grounding_ratio: float = Field(0.34, ge=0.0, le=1.0)

    # --------------------------------------------------------------------- LLM
    # Provider behavior is explicit; all supported providers use an
    # OpenAI-compatible chat-completions API.
    llm_provider: Literal["groq", "vllm", "openai-compatible", "echo"] = "openai-compatible"
    llm_base_url: str = ""
    llm_model: str = ""
   
    llm_api_key: str = Field("", validation_alias=AliasChoices("llm_api_key", "groq_api_key"))
    # Tokens reserved for the answer when assembling prompt context
    context_answer_reserve_tokens: int = Field(
   
    llm_max_completion_tokens: int | None = Field(None, ge=1)
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(120.0, gt=0.0)
    # Maximum input/context budget used by the application
    llm_context_window: int = Field(8192, ge=1024)
    # Retry transient model-provider failures
    llm_max_retries: int = Field(3, ge=0, le=10)
    llm_retry_max_wait_s: float = Field(30.0, gt=0.0)
    # Minimum delay between model requests;
    llm_min_call_interval_s: float = Field(0.0, ge=0.0)
    
    llm_extra_body: dict = Field(default_factory=dict)
    llm_price_input_per_mtok: float | None = Field(None, ge=0.0)
    llm_price_output_per_mtok: float | None = Field(None, ge=0.0)

    allow_test_doubles: bool = False

    # ---------------------------------------------------------------- AirLabs
    airlabs_key: str = ""
    airlabs_base: str = "https://airlabs.co/api/v9"
    airlabs_monthly_quota: int = Field(1000, ge=0)
    airlabs_cache_ttl_s: int = Field(60, ge=0)
    # Stop live calls when the remaining monthly quota reaches this limit
    airlabs_reserve: int = Field(100, ge=0)
    # Use recorded flight responses for tests and deterministic replay only
    use_flight_fixtures: bool = False

    # ------------------------------------------------------------------- app
    context_token_budget: int = Field(6000, ge=1)
    max_validation_retries: int = Field(1, ge=0, le=5)
  
    enable_stream_endpoint: bool = False
    log_level: str = "INFO"

    # -------------------------------------------------------- public exposure
   
    conversation_secret: str = ""
    # Requests allowed per client per minute.
    rate_limit_per_minute: int = Field(30, ge=1)
    trust_forwarded_for: bool = False

    @model_validator(mode="after")
    def _check_relationships(self) -> Settings:
        problems = []
        if self.final_k > self.candidates_k:
            problems.append(f"final_k ({self.final_k}) exceeds candidates_k ({self.candidates_k})")
        if self.balance_sources and self.min_government_sources + self.min_airline_sources > self.final_k:
            problems.append(
                f"min_government_sources + min_airline_sources "
                f"({self.min_government_sources} + {self.min_airline_sources}) exceeds final_k "
                f"({self.final_k}): the quotas cannot fit in the final context")
        if not (self.use_dense or self.use_sparse):
            problems.append("use_dense and use_sparse are both false: nothing would be retrieved")
        if self.airlabs_monthly_quota and self.airlabs_reserve >= self.airlabs_monthly_quota:
            problems.append(
                f"airlabs_reserve ({self.airlabs_reserve}) must be below airlabs_monthly_quota "
                f"({self.airlabs_monthly_quota}), or every live lookup is refused")
        if self.context_answer_reserve_tokens + 1024 > self.llm_context_window:
            problems.append(
                f"context_answer_reserve_tokens ({self.context_answer_reserve_tokens}) leaves "
                f"under 1024 tokens of llm_context_window ({self.llm_context_window}) for the "
                "prompt and sources")
        if (self.llm_max_completion_tokens is not None
                and self.llm_max_completion_tokens < self.context_answer_reserve_tokens):
            problems.append(
                f"llm_max_completion_tokens ({self.llm_max_completion_tokens}) is below "
                f"context_answer_reserve_tokens ({self.context_answer_reserve_tokens}), the "
                "answer reserve it must hold")
        if (self.llm_max_completion_tokens is None
                and {"reasoning_effort", "reasoning"} & set(self.llm_extra_body)):
            problems.append(
                "llm_extra_body configures reasoning but LLM_MAX_COMPLETION_TOKENS is unset: set it "
                "explicitly to the completion cap the model was evaluated with (it would otherwise "
                f"fall back to the {self.context_answer_reserve_tokens}-token answer reserve)")
        if self.llm_provider == "groq" and self.llm_api_key in ("", "not-needed-for-local"):
            problems.append("llm_provider is groq but LLM_API_KEY is not set")
        if problems:
            raise ValueError("invalid settings: " + "; ".join(problems))
        return self

    @property
    def llm_max_tokens(self) -> int:
        """Former name of context_answer_reserve_tokens ."""
        return self.context_answer_reserve_tokens

    @property
    def completion_cap_tokens(self) -> int:
        """The completion cap actually sent to the provider."""
        return self.llm_max_completion_tokens or self.context_answer_reserve_tokens

    def test_double_problems(self) -> list[str]:
        """
        Return unsafe test doubles configured for real serving/indexing
        """
        if self.allow_test_doubles:
            return []
        out = []
        if self.embedder == "hash":
            out.append("EMBEDDER=hash is a test double (hash embeddings are not semantic)")
        if self.llm_provider == "echo" or self.llm_base_url in ("echo", "none"):
            out.append("the echo generator is a test double; set LLM_PROVIDER and LLM_BASE_URL "
                       "to a real model endpoint")
        if self.use_flight_fixtures:
            out.append("USE_FLIGHT_FIXTURES=true serves recorded past flights as if they were live")
        return out


def revalidate(settings: Settings) -> Settings:
    """
    Re-run validation on a derived Settings object without re-reading .env
    """
    return Settings(_env_file=None, **settings.model_dump())


@lru_cache
def get_settings() -> Settings:
    return Settings()
