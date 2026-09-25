"""
Core data types.

Every stage of the RAG pipeline hands data to the next stage. 

These are plain dataclasses (not pydantic) because they are internal and never
cross an HTTP boundary. The API request/response models in api.py use pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# "clarify": the answer is a question back to the passenger (e.g. their flight
# number), because the departure airport decides which law applies.
Intent = Literal["policy", "status", "hybrid", "clarify", "unsupported"]

# What the pipeline actually did with a turn. Evaluation and metrics read this,
# never the prose: a decline and an answer can share words.
#   answered              generated, validated answer
#   abstained             generated, validated, and the model declined (Rule 7)
#   clarified             asked the passenger for missing route/flight details
#   followup              a recognised short follow-up, answered from the PREVIOUS
#                         turn's answer with NO model call (followups.py). Distinct
#                         from "answered" on purpose: nothing was generated, nothing
#                         was retrieved, and no source was consulted that the
#                         passenger had not already been shown.
#   gated                 declined BEFORE generation: retrieval too weak
#   declined_unsupported  carrier not covered and nothing carrier-independent to say
#   validation_failed     generated, still invalid after retries: safe message shown instead
#   llm_error             the model call failed: safe message shown instead
Outcome = Literal["answered", "abstained", "clarified", "gated", "declined_unsupported",
                  "validation_failed", "llm_error", "followup"]

# Where flight data came from. Only "live" and "cache" describe the flight as it is now.
FlightSource = Literal["live", "cache", "fixture", "synthetic"]


DocType = Literal["regulation", "guidance", "contract", "service_plan", "policy", ""]


@dataclass(frozen=True)
class Section:
    """
    One node in a document's structure tree.

    eCFR XML gives this hierarchy for free. Airline PDFs require heading
    detection to infer it. Both parse into this neutral shape so the chunker
    does not need to change — only the parser does.
    """

    doc_id: str          # e.g. "aa-customer-service-plan"
    doc_title: str       # e.g. "American Airlines Customer Service Plan"
    publisher: str       # e.g. "American Airlines" or "US DOT"
    jurisdiction: str    # "US" | "EU" | "UK"
    section_id: str      # e.g. "delays-cancellations"
    heading: str
    path: tuple[str, ...]
    text: str
    source_url: str
    airline_iata: str = ""    # "AA", "DL", "UA", "WN"; empty for government docs
    doc_type: DocType = ""    # regulation | guidance | contract | service_plan | policy
    effective_date: str | None = None

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.path)

    @property
    def source_class(self) -> str:
        return "airline" if self.airline_iata else "government"


@dataclass
class Chunk:
    """
    A retrievable unit. Gets embedded, stored, searched, reranked, and cited.

    `text` is the raw clause body. `embed_text` includes the breadcrumb prefix.
    Keeping them separate matters: the breadcrumb influences the vector but is
    not duplicated in the LLM's context window.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    publisher: str
    jurisdiction: str
    section_id: str
    breadcrumb: str
    text: str
    embed_text: str
    source_url: str
    airline_iata: str = ""
    doc_type: DocType = ""
    effective_date: str | None = None
    token_estimate: int = 0

    score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    # Set by retrieval.balance_by_source_class when this chunk holds the law of a
    # regime that GOVERNS the route and nothing else in the context does
    guaranteed: bool = False

    @property
    def source_class(self) -> str:
        """
        Which of the two evidence families this chunk belongs to.

        """
        return "airline" if self.airline_iata else "government"


@dataclass
class FlightStatus:
    """
    Normalised subset of an AirLabs API response.

    NOTE the absence of any `delay_cause` field. AirLabs does not report cause,
    and that absence is the single most important fact about this data source.
    See docs/adr/0001.
    """

    flight_iata: str
    airline_iata: str | None
    status: str | None              # scheduled | active | landed | cancelled
    dep_iata: str | None
    dep_time: str | None            # scheduled departure (local)
    dep_estimated: str | None       # updated departure (local)
    dep_delayed: int | None         # departure delay minutes
    arr_iata: str | None
    arr_time: str | None            # scheduled arrival (local)
    arr_estimated: str | None       # updated arrival (local)
    arr_delayed: int | None         # arrival delay minutes
    delayed: int | None             # overall delay minutes
    fetched_at: str = ""
    from_cache: bool = False
    source: FlightSource = "live"
    # Codeshare: the airline actually flying it, when the data source reports one
    # that differs from the marketing carrier in the flight number.
    operating_airline_iata: str | None = None
    operating_flight_iata: str | None = None

    @property
    def flight_date(self) -> str | None:
        """YYYY-MM-DD of the scheduled departure, when the record carries one."""
        d = (self.dep_time or "")[:10]
        return d if len(d) == 10 and d[4] == "-" and d[7] == "-" else None

    @property
    def is_codeshare(self) -> bool:
        return bool(self.operating_airline_iata and self.airline_iata
                    and self.operating_airline_iata.upper() != self.airline_iata.upper())

    @property
    def worst_delay_min(self) -> int | None:
        vals = [d for d in (self.dep_delayed, self.arr_delayed, self.delayed) if d is not None]
        return max(vals) if vals else None


@dataclass
class Citation:
    marker: str
    chunk_id: str
    doc_title: str
    section_id: str
    source_url: str


@dataclass
class Answer:
    question: str
    intent: Intent
    text: str
    citations: list[Citation] = field(default_factory=list)
    context_chunks: list[Chunk] = field(default_factory=list)
    live_data: FlightStatus | None = None
    validation_failures: list[str] = field(default_factory=list)
    retry_count: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    clarification: bool = False     # text is a question for the passenger, not an answer
    assumption: str | None = None   # route assumption the answer opens with, if any
    unsupported_airline: str | None = None  # IATA code of a carrier this assistant does not cover
    outcome: Outcome = "answered"
    # True only when the model was actually called (intent alone does not say so).
    generation_attempted: bool = False
    # The exact context the model saw (generation.BuiltContext), so evaluation
    # scores what was generated from instead of rebuilding (and re-ordering) it.
    built_context: object | None = None
    prompt_sha256: str | None = None
    # Diagnostics for logs and evaluation, never shown to the passenger (for
    # example the rejected answer text after a validation failure).
    diagnostics: dict = field(default_factory=dict)
