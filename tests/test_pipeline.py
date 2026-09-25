"""
Test suite.

WHAT IS TESTED AND WHAT IS NOT
-------------------------------
Tested here: the LOGIC that is easy to get subtly wrong and hard to notice —
chunk boundaries, rank fusion arithmetic, citation validation, intent routing,
metric bounds. All of it runs offline in under a second with no Postgres, no
network, and no model download.

Not tested here: the Postgres SQL and the real models. The SQL is covered by
tests/test_postgres.py, which runs only when FDR_TEST_PG_DSN points at a
disposable database; the real models by the two evaluation stages. Mixing unit
and integration tests makes the fast ones slow and the slow ones flaky.

"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Kept in the working copy but git-ignored, so a public clone does not have it.
# Doc checks read it when present and skip it when absent; every other file they
# name is still required.
LOCAL_ONLY_DOCS = {ROOT / "CLAUDE.md"}
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "scripts"))

from flight_delay import followups  # noqa: E402
from flight_delay.config import Settings  # noqa: E402
from flight_delay.embeddings import HashEmbedder, tokenize  # noqa: E402
from flight_delay.generation import build_context, validate_answer  # noqa: E402
from flight_delay.ingest import (  # noqa: E402
    chunk_section,
    estimate_tokens,
    parse_ecfr_xml,
    sections_from_markdown,
    sections_from_plaintext,
)
from flight_delay.jurisdiction import detect_jurisdictions  # noqa: E402
from flight_delay.models import Chunk, FlightStatus  # noqa: E402
from flight_delay.pipeline import classify  # noqa: E402
from flight_delay.retrieval import (  # noqa: E402
    balance_by_source_class,
    reciprocal_rank_fusion,
)
from flight_delay.store import MemoryStore  # noqa: E402
from flight_delay.tools import FileQuotaCounter, extract_flight_number  # noqa: E402

FIXTURE = ROOT / "data" / "fixtures" / "synthetic_part250.xml"


# ==========================================================================
# Parsing & chunking
# ==========================================================================


def _sections():
    return parse_ecfr_xml(
        FIXTURE.read_bytes(),
        doc_id="cfr-250",
        doc_title="14 CFR Part 250",
        source_url="https://example.test/250",
        effective_date="2026-01-01",
    )


def test_parser_finds_all_sections():
    secs = _sections()
    ids = {s.section_id for s in secs}
    assert ids == {"250.1", "250.5", "250.6", "250.9", "250.10"}


def test_breadcrumb_does_not_duplicate_section_number():
    """Regression: eCFR HEAD already contains '§ 250.5', so naive prefixing
    produced '§ 250.5 § 250.5 Amount of...' in every chunk and every citation."""
    for s in _sections():
        assert s.breadcrumb.count(f"§ {s.section_id}") <= 1


def test_chunks_never_span_sections():
    """The invariant that makes citations trustworthy."""
    for sec in _sections():
        for ch in chunk_section(sec, target_tokens=100):
            assert ch.section_id == sec.section_id
            assert ch.text in sec.text or all(
                part.strip() in sec.text for part in ch.text.split("\n\n")
            )


def test_breadcrumb_is_in_embed_text_but_not_in_text():
    """The breadcrumb must influence the VECTOR without polluting the LLM's
    context window or the text the user is shown."""
    sec = _sections()[1]
    ch = chunk_section(sec, breadcrumb=True)[0]
    assert "14 CFR Part 250" in ch.embed_text
    assert "14 CFR Part 250" not in ch.text
    assert ch.text in ch.embed_text


def test_breadcrumb_can_be_disabled_for_ablation():
    sec = _sections()[1]
    off = chunk_section(sec, breadcrumb=False)[0]
    assert off.embed_text == off.text


def test_chunk_ids_are_stable_across_runs():
    """Content-addressed ids are what make re-indexing idempotent."""
    sec = _sections()[1]
    a = [c.chunk_id for c in chunk_section(sec)]
    b = [c.chunk_id for c in chunk_section(sec)]
    assert a == b


def test_chunks_respect_the_token_budget():
    for sec in _sections():
        for ch in chunk_section(sec, target_tokens=120):
            assert ch.token_estimate <= 120 * 1.4 + 30


def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 100)


# ==========================================================================
# Fusion
# ==========================================================================


def test_rrf_rewards_documents_found_by_both_retrievers():
    """The core property of RRF: agreement between retrievers beats a single
    retriever's top hit."""
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("c", 50.0), ("b", 40.0), ("z", 30.0)]
    fused = dict(reciprocal_rank_fusion([dense, sparse], k=60))
    # b is rank 2 in both; a is rank 1 in one and absent from the other.
    assert fused["b"] > fused["a"]
    assert fused["c"] > fused["a"]


def test_rrf_ignores_score_magnitude():
    """Scores from BM25 and cosine are not comparable. RRF must not care."""
    d1 = [("a", 0.01), ("b", 0.009)]
    d2 = [("a", 9999.0), ("b", 9998.0)]
    assert reciprocal_rank_fusion([d1]) == reciprocal_rank_fusion([d2])


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# ==========================================================================
# Retrieval plumbing
# ==========================================================================


def _mem_store():
    secs = _sections()
    chunks = [c for s in secs for c in chunk_section(s, target_tokens=150)]
    emb = HashEmbedder(256)
    st = MemoryStore()
    st.upsert(chunks, emb.embed_documents([c.embed_text for c in chunks]))
    return st, emb, chunks


def test_bm25_finds_exact_numeric_tokens():
    """The reason sparse retrieval is in the stack at all: embeddings smear
    numbers, BM25 does not."""
    st, _, _ = _mem_store()
    hits = st.sparse_search("stopover four hours", 5)
    assert hits
    assert any(st.chunks[cid].section_id == "250.1" for cid, _ in hits)


def test_metadata_filter_excludes_non_matching_jurisdiction():
    st, emb, _ = _mem_store()
    q = emb.embed_query("compensation")
    assert st.dense_search(q, 10, {"jurisdiction": "EU"}) == []
    assert st.dense_search(q, 10, {"jurisdiction": "US"}) != []


def test_reingest_is_idempotent():
    st, emb, chunks = _mem_store()
    before = st.count()
    st.upsert(chunks, emb.embed_documents([c.embed_text for c in chunks]))
    assert st.count() == before


# ==========================================================================
# Context assembly
# ==========================================================================


def _chunk(cid: str, text: str, tokens: int = 30, airline: str = "",
           doc_type: str = "regulation") -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d", doc_title="14 CFR Part 250", publisher="DOT",
        jurisdiction="US", section_id="250.5", breadcrumb="Part 250 > 250.5",
        text=text, embed_text=text, source_url="u", airline_iata=airline,
        doc_type=doc_type, effective_date="2026-01-01", token_estimate=tokens,
    )


def test_best_chunk_is_placed_last_in_context():
    """'Lost in the middle': attention is strongest at the end of the context,
    so the top-ranked chunk goes closest to the question."""
    chunks = [_chunk("c1", "BEST"), _chunk("c2", "SECOND"), _chunk("c3", "THIRD")]
    ctx = build_context(chunks, None)
    assert ctx.text.index("BEST") > ctx.text.index("THIRD")
    assert ctx.source_map["S3"].chunk_id == "c1"


def test_context_respects_the_token_budget():
    chunks = [_chunk(f"c{i}", "x" * 4000, tokens=1000) for i in range(20)]
    ctx = build_context(chunks, None, token_budget=3000, reserve_for_answer=500)
    assert len(ctx.used_chunks) < 20
    assert ctx.estimated_tokens <= 3000


def test_flight_block_states_that_cause_is_unavailable():
    """The single most important product behaviour: never let the model imply
    it knows why a flight was delayed."""
    fs = FlightStatus(
        flight_iata="UA2402", airline_iata="UA", status="active",
        dep_iata="LAX", dep_time=None, dep_estimated=None, dep_delayed=217,
        arr_iata="BOS", arr_time=None, arr_estimated=None, arr_delayed=198,
        delayed=198,
    )
    ctx = build_context([_chunk("c1", "text")], fs)
    assert "NOT REPORTED" in ctx.text
    assert "217" in ctx.text
    assert fs.worst_delay_min == 217


# ==========================================================================
# Source balancing  (law + airline policy must both reach the answer)
# ==========================================================================


def _gov(cid: str) -> Chunk:
    return _chunk(cid, f"regulation text {cid}", doc_type="regulation")


def _air(cid: str) -> Chunk:
    return _chunk(cid, f"airline text {cid}", airline="AA", doc_type="contract")


def test_source_class_splits_law_from_airline_policy():
    assert _gov("g1").source_class == "government"
    assert _air("a1").source_class == "airline"


def test_balancing_pulls_in_airline_text_when_law_dominates_ranking():
    """The failure this exists to prevent: a compensation question ranks six
    regulation chunks at the top, and the answer never mentions that the airline
    promises more than the legal floor."""
    ranked = [_gov(f"g{i}") for i in range(6)] + [_air(f"a{i}") for i in range(4)]
    out = balance_by_source_class(ranked, 6, min_government=2, min_airline=2)

    assert len(out) == 6
    assert sum(1 for c in out if c.source_class == "airline") >= 2
    assert sum(1 for c in out if c.source_class == "government") >= 2


def test_balancing_pulls_in_law_when_airline_text_dominates_ranking():
    ranked = [_air(f"a{i}") for i in range(6)] + [_gov(f"g{i}") for i in range(4)]
    out = balance_by_source_class(ranked, 6, min_government=2, min_airline=2)
    assert sum(1 for c in out if c.source_class == "government") >= 2


def test_balancing_preserves_best_first_ordering():
    """build_context puts the LAST chunk closest to the question, so a balanced
    result that lost its ordering would silently bury the best evidence."""
    ranked = [_gov(f"g{i}") for i in range(6)] + [_air(f"a{i}") for i in range(4)]
    out = balance_by_source_class(ranked, 6, min_government=2, min_airline=2)
    positions = [ranked.index(c) for c in out]
    assert positions == sorted(positions)


def test_balancing_does_not_invent_slots_when_one_family_is_absent():
    """A corpus with no airline text for this query must still return top_k
    results, not a short list padded with nothing."""
    ranked = [_gov(f"g{i}") for i in range(8)]
    out = balance_by_source_class(ranked, 5, min_government=2, min_airline=2)
    assert len(out) == 5
    assert all(c.source_class == "government" for c in out)


def test_balancing_never_exceeds_top_k():
    ranked = [_gov(f"g{i}") for i in range(5)] + [_air(f"a{i}") for i in range(5)]
    assert len(balance_by_source_class(ranked, 3, min_government=2, min_airline=2)) == 3


def _eu_law(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="eu261", doc_title="EU261", publisher="EU",
        jurisdiction="EU", section_id="7", breadcrumb="Article 7",
        text="fixed compensation", embed_text="fixed compensation",
        source_url="u", doc_type="regulation", token_estimate=10,
    )


def test_required_jurisdiction_gets_a_slot_even_when_outranked():
    """The Paris failure: US law out-ranks EU261 on wording, so without a
    reserved slot an EU-departure question is answered purely from US law —
    which says no delay compensation is owed, when EU261 says up to EUR 600."""
    ranked = [_gov(f"g{i}") for i in range(6)] + [_air("a1"), _eu_law("eu1")]
    out = balance_by_source_class(
        ranked, 6, min_government=2, min_airline=2, require_jurisdictions=("EU",)
    )
    assert any(c.jurisdiction == "EU" for c in out)


def test_airline_contract_filed_under_us_does_not_satisfy_us_law():
    """UK->US: airline contracts carry jurisdiction US, and used to count as the US
    regime's seat, so contexts reached the model with UK261 and no US law."""
    uk = [_eu_law(f"uk{i}") for i in range(2)]
    for c in uk:
        c.jurisdiction = "UK"
    us_air = _air("a1")
    us_air.jurisdiction = "US"
    ranked = uk + [us_air, _air("a2"), _air("a3"), _air("a4"), _gov("us-law")]
    out = balance_by_source_class(
        ranked, 6, min_government=2, min_airline=2, require_jurisdictions=("UK", "US")
    )
    assert any(c.source_class == "government" and c.jurisdiction == "US" for c in out)


def test_required_jurisdiction_is_skipped_when_corpus_has_none():
    ranked = [_gov(f"g{i}") for i in range(4)] + [_air("a1")]
    out = balance_by_source_class(
        ranked, 4, min_government=2, min_airline=1, require_jurisdictions=("EU",)
    )
    assert len(out) == 4


def test_required_jurisdiction_does_not_duplicate_when_already_present():
    ranked = [_eu_law("eu1"), _gov("g1"), _air("a1"), _gov("g2")]
    out = balance_by_source_class(
        ranked, 4, min_government=2, min_airline=1, require_jurisdictions=("EU",)
    )
    assert len({c.chunk_id for c in out}) == len(out)


def test_context_labels_law_and_airline_sources_differently():
    """The model cannot keep entitlement separate from a carrier's promise if
    the context does not tell it which is which."""
    ctx = build_context([_gov("g1"), _air("a1")], None)
    assert "LAW" in ctx.text
    assert "AIRLINE" in ctx.text


# ==========================================================================
# Markdown & plaintext structure detection
# ==========================================================================


def test_markdown_headings_become_sections_with_hierarchy():
    md = (
        "# Delta Policy\n\nIntro paragraph that is long enough to keep.\n\n"
        "## Requesting a Refund\n\nRefund body text here.\n\n"
        "### Refund Restrictions\n\nRestriction body text here.\n"
    )
    secs = sections_from_markdown(md, doc_id="d", doc_title="Delta", publisher="DL")
    headings = [s.heading for s in secs]
    assert "Requesting a Refund" in headings
    assert "Refund Restrictions" in headings
    # The nested heading keeps its parent in the breadcrumb.
    nested = next(s for s in secs if s.heading == "Refund Restrictions")
    assert "Requesting a Refund" in nested.breadcrumb


def test_plaintext_allcaps_headings_are_detected():
    """Regression: scraped exports used ALL-CAPS headings, and the old parser
    recognised only 'Rule/Article/Section N', collapsing the whole document into
    one 'Preamble' section with no usable citation."""
    txt = (
        "OVERVIEW\n--------\nSome overview text for the document.\n\n"
        "COMPENSATION AMOUNTS\n--------------------\nYou may be owed 600 EUR.\n"
    )
    secs = sections_from_plaintext(txt, doc_id="d", doc_title="EU", publisher="EU")
    headings = [s.heading for s in secs]
    assert "OVERVIEW" in headings
    assert "COMPENSATION AMOUNTS" in headings


def test_doc_type_survives_into_chunks():
    secs = sections_from_markdown(
        "## A Heading\n\nSome body text long enough to survive the filter.\n",
        doc_id="d", doc_title="T", publisher="AA", airline_iata="AA",
        doc_type="service_plan",
    )
    ch = chunk_section(secs[0])[0]
    assert ch.doc_type == "service_plan"
    assert ch.source_class == "airline"


# ==========================================================================
# Validation  (the safety net)
# ==========================================================================


def _ctx_one_source():
    return build_context([_chunk("c1", "Compensation is 400 percent of the fare.")], None)


def test_valid_cited_answer_passes():
    r = validate_answer("You are owed 400 percent of the fare [S1].", _ctx_one_source())
    assert r.ok and len(r.citations) == 1


def test_fabricated_citation_is_caught():
    """The most dangerous failure mode: a confident answer citing a source that
    was never retrieved."""
    r = validate_answer("You are owed compensation under Rule 12 [S7].", _ctx_one_source())
    assert not r.ok
    assert "S7" in r.unknown_markers


def test_uncited_factual_claim_is_caught():
    r = validate_answer(
        "The airline must pay you 400 percent of your fare and provide a hotel.",
        _ctx_one_source(),
    )
    assert not r.ok


def test_next_step_lines_are_claims_once_they_name_money():
    """Session 19, the real gpt-oss-20b answer to "My United flight from London
    Heathrow to Chicago was delayed 5 hours": the LAW was right (UK261, GBP 520) and
    cited; the two rejected sentences were both closing ACTION lines. Rule 3 now
    tells the model to re-cite an action line that names money, and to start such a
    line with the verb. These are the exact sentences the run rejected."""
    ctx = _ctx_one_source()
    claim = "You are owed 400 percent of the fare [S1]."   # the cited part, as the model had it

    # 1. an instruction is fine uncited - until it names an amount
    assert validate_answer(f"{claim} Ask United for their claim form.", ctx).ok
    assert not validate_answer(
        f"{claim} Ask United for the GBP 520 compensation claim form.", ctx).ok
    # ...and carrying the marker fixes it
    assert validate_answer(
        f"{claim} Ask United for the GBP 520 compensation claim form [S1].", ctx).ok

    # 2. "let them ..." is not an instruction the validator recognises, so it reads
    #    as a claim; the same advice as a direct imperative passes
    soft = (f"{claim} If you are unsure whether the delay was within United's control, "
            "let them confirm the cause so you can claim the full compensation.")
    assert not validate_answer(soft, ctx).ok
    firm = (f"{claim} If you are unsure whether the delay was within United's control, "
            "ask them to confirm the cause.")
    assert validate_answer(firm, ctx).ok


def test_system_prompt_warns_about_both_citation_traps():
    """The two rules added in Session 19 after they were measured failing. Kept as a
    test because the file is the single source of the prompt and is easy to trim."""
    from flight_delay.generation import SYSTEM_PROMPT

    assert "WHICH LAW APPLIES IS ITSELF A CLAIM" in SYSTEM_PROMPT
    assert "THE OPENING SENTENCE COUNTS" in SYSTEM_PROMPT
    assert "NEXT STEPS ARE CHECKED TOO" in SYSTEM_PROMPT
    assert "DECLINE AND STOP" in SYSTEM_PROMPT


def test_explicit_abstention_is_accepted_not_penalised():
    """A system that correctly refuses must not be scored as failing, or we
    would be optimising it toward confident guessing."""
    r = validate_answer(
        "The provided sources do not cover baggage fees. You would need the "
        "airline's contract of carriage.",
        _ctx_one_source(),
    )
    assert r.ok and r.is_abstention


def test_partially_cited_answer_is_caught():
    r = validate_answer(
        "Compensation is 400 percent [S1]. The airline shall also pay for meals.",
        _ctx_one_source(),
    )
    assert not r.ok


# ==========================================================================
# Routing
# ==========================================================================


@pytest.mark.parametrize(
    "question,intent",
    [
        ("Am I entitled to a hotel if ATC caused the delay?", "policy"),
        ("How much compensation for denied boarding?", "policy"),
        ("Can I bring a snowboard?", "policy"),
        ("Is UA2402 delayed today?", "status"),
        ("BA 117 status", "status"),
        ("My flight UA2402 is delayed, what am I owed?", "hybrid"),
    ],
)
def test_router(question, intent):
    assert classify(question)[0] == intent


@pytest.mark.parametrize("question", [
    "my delta flight in cacelled. what can i do?",
    "my flight got cancled",
    "our flight was delyed 3 hours",
])
def test_misspelled_disruption_still_asks_for_the_route(question):
    """A typo used to skip the clarifying question and answer on unstated US rules."""
    assert detect_jurisdictions(question).needs_clarification


def test_follow_up_question_needs_no_citation_but_a_claim_in_one_does():
    ok = validate_answer(
        "You can get a full refund if you choose not to travel [S1]. "
        "Did Delta say why it was cancelled - weather, or a crew or mechanical problem?",
        _ctx_one_source())
    assert ok.ok, ok.failures
    bad = validate_answer(
        "You can get a full refund [S1]. Did you know you are entitled to compensation of 600 euros?",
        _ctx_one_source())
    assert not bad.ok


@pytest.mark.parametrize(
    "question",
    [
        "My AA2359 flight was cancelled. What compensation can I get?",
        "Can I get a refund for my delayed flight?",
        "I was denied boarding. What are my rights?",
        "My connection was missed after a delay",
        "Is DL 9902 delayed, and do I get anything?",
        "I'm on AA 9903. What are my options?",
    ],
)
def test_router_rights_questions_are_not_status_only(question):
    assert classify(question)[0] in ("policy", "hybrid")


@pytest.mark.parametrize(
    "question",
    [
        # Each hits exactly one inflected rights term the old whole-stem regex missed.
        "AA2359 compensation?",
        "Will I be compensated for AA2359?",
        "AA2359 cancelled",
        "AA2359 canceled",
        "AA2359 cancellation",
        "Cancel AA2359",
        "AA2359 refunds",
        "AA2359 refund",
        "Denied boarding on AA2359",
        "AA2359 was overbooked",
        "AA2359 overbooking",
        "AA2359 rights",
        "Missed connection off AA2359",
        "My connection was missed after AA2359 was delayed",
        "My AA2359 flight was cancelled. What compensation can I get?",
    ],
)
def test_router_inflected_rights_terms_with_flight_are_hybrid(question):
    assert classify(question) == ("hybrid", "AA2359")


@pytest.mark.parametrize(
    "question",
    ["What is the status of AA2359?", "Is AA2359 delayed?", "AA2359 delays", "AA2359"],
)
def test_router_status_questions_stay_status_only(question):
    assert classify(question) == ("status", "AA2359")


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What does United owe me for a cancelled flight?", "UA"),
        ("Delta delayed my flight 5 hours", "DL"),
        ("american airlines lost my bag", "AA"),
        ("Southwest oversold my flight", "WN"),
        ("How much compensation for denied boarding?", None),
    ],
)
def test_airline_detection(question, expected):
    from flight_delay.pipeline import detect_airline

    assert detect_airline(question) == expected


def test_flight_number_beats_airline_name_in_the_text():
    """'I'm on UA2402 instead of my Delta flight' is about United."""
    from flight_delay.pipeline import detect_airline

    assert detect_airline("I rebooked off my Delta flight onto UA2402", "UA2402") == "UA"


def test_airline_scope_keeps_regulations_but_drops_other_carriers():
    """The whole point of airline_scope: narrow the airline evidence without
    discarding the law, which carries no airline code."""
    store = MemoryStore()
    emb = HashEmbedder(256)
    chunks = [
        _chunk("g1", "regulation about denied boarding compensation"),
        _chunk("ua1", "united policy on denied boarding", airline="UA", doc_type="contract"),
        _chunk("dl1", "delta policy on denied boarding", airline="DL", doc_type="contract"),
    ]
    store.upsert(chunks, emb.embed_documents([c.embed_text for c in chunks]))

    hits = store.sparse_search("denied boarding", 10, {"airline_scope": "UA"})
    ids = {cid for cid, _ in hits}
    assert "dl1" not in ids          # other carrier excluded
    assert "g1" in ids               # regulation kept
    assert "ua1" in ids


# ==========================================================================
# Jurisdiction routing
# ==========================================================================


# The applicability matrix for the four US carriers in this corpus. EU261 and
# UK261 bind a non-EU/UK carrier only on DEPARTURE from their territory, so the
# origin airport decides the whole question.
@pytest.mark.parametrize(
    "dep,arr,governing",
    [
        ("LHR", "JFK", ("US", "UK")),   # UK -> US : UK261 + DOT
        ("CDG", "JFK", ("US", "EU")),   # EU -> US : EU261 + DOT
        ("LHR", "CDG", ("UK",)),        # UK -> EU : UK261 only, never touches US
        ("CDG", "LHR", ("EU",)),        # EU -> UK : EU261 only, never touches US
        ("JFK", "LHR", ("US",)),        # US -> UK : no UK261, DOT rules only
        ("JFK", "CDG", ("US",)),        # US -> EU : no EU261, DOT rules only
        ("ATL", "DFW", ("US",)),        # domestic
    ],
)
def test_applicability_matrix_from_live_flight_data(dep, arr, governing):
    route = detect_jurisdictions("my flight was delayed", dep_iata=dep, arr_iata=arr)
    assert route.governing == governing


@pytest.mark.parametrize(
    "dep,arr",
    [("CDG", "LHR"), ("LHR", "CDG"), ("CDG", "FRA"), ("LHR", "EDI")],
)
def test_intra_europe_routes_carry_no_us_regulation(dep, arr):
    """An itinerary that starts and ends in Europe never touches the US, so no
    US DOT rule is in play even though the operating carrier is American."""
    route = detect_jurisdictions("delayed 5 hours", dep_iata=dep, arr_iata=arr)
    assert "US" not in route.governing
    assert "US" not in route.scope
    assert route.touches_us is False


def test_intra_europe_detected_from_question_text():
    route = detect_jurisdictions("my flight from Paris to London Heathrow was cancelled")
    assert route.governing == ("EU",)
    assert "US" not in route.scope


def test_unknown_destination_keeps_us_retrievable_but_not_governing():
    """Paris -> ?: EU261 governs (EU departure). An unknown end is not the US,
    but may be: US DOT stays in scope so refund rules are not lost from
    retrieval, and is not governing until a US endpoint is known or assumed.
    The passenger is asked where the flight was going."""
    route = detect_jurisdictions("my flight from Paris was delayed")
    assert route.touches_us is False
    assert route.governing == ("EU",)
    assert "US" in route.scope
    assert route.needs_clarification is True


def test_foreign_regime_is_retrievable_even_when_it_does_not_govern():
    """US->Paris must still be able to cite EU261's scope article to explain
    why no compensation is owed — retrievable, but not guaranteed a slot."""
    route = detect_jurisdictions("delayed", dep_iata="JFK", arr_iata="CDG")
    assert "EU" in route.scope
    assert "EU" not in route.governing


@pytest.mark.parametrize(
    "question,governing",
    [
        ("My flight from Paris to New York was delayed", ("US", "EU")),
        ("My flight from New York to Paris was delayed", ("US",)),
        ("Paris to New York, delayed 4 hours", ("US", "EU")),
        ("Flight out of LHR cancelled", ("UK",)),               # destination unknown
        ("My flight to London Heathrow was cancelled", ()),     # departure unknown
        ("Departing Frankfurt, delayed 5 hours", ("EU",)),
        ("My flight from Dallas to Denver was delayed", ("US",)),
        ("How much for denied boarding?", ()),                  # no route at all
    ],
)
def test_direction_is_read_from_the_question_text(question, governing):
    assert detect_jurisdictions(question).governing == governing


def test_nearest_preposition_wins():
    """'from New York to Paris' — both prepositions precede 'Paris', and only
    the nearest one ('to') describes it."""
    route = detect_jurisdictions("my flight from New York to Paris was delayed")
    assert route.governing == ("US",)
    assert "EU" in route.scope


def test_ambiguous_direction_scopes_but_does_not_guarantee():
    """'my Paris flight' does not say which way it was going: EU261 is
    retrievable but not guaranteed until the passenger answers."""
    route = detect_jurisdictions("my Paris flight was cancelled, what am I owed?")
    assert "EU" in route.scope
    assert "EU" not in route.governing
    assert route.direction_known is False
    assert route.needs_clarification is True


@pytest.mark.parametrize(
    "question",
    [
        "my delta flight was delayed 5 hours",               # no place at all
        "my paris flight was delayed what do i do?",
        "My flight to London Heathrow was cancelled",      # destination alone is not a departure
        "American denied me boarding on the Dublin–Chicago flight",  # a dash is not a direction
        "my manchester to newark flight delayed",          # Manchester UK or NH?
        "My flight from Nantes to Chicago was cancelled",  # city in no table
    ],
)
def test_unknown_departure_needs_clarification(question):
    """The departure airport decides EU261/UK261. When the passenger has not
    said where they left from, nothing is assumed before asking."""
    route = detect_jurisdictions(question)
    assert route.needs_clarification is True
    assert route.origin_region is None
    assert not {"EU", "UK"} & set(route.governing)   # these turn on the unknown departure
    assert "US" in route.scope                       # an unknown end may be a US airport


@pytest.mark.parametrize(
    "question,governing",
    [
        ("American cancelled my Chicago to Paris flight", ("US",)),
        ("Delta downgraded me from Delta One to economy on my Paris to Atlanta flight", ("US", "EU")),
        ("Fog at Heathrow delayed my Delta flight to Atlanta by 5 hours", ("US", "UK")),
        ("Snow in Amsterdam delayed my Delta flight to Detroit", ("US", "EU")),
        ("A winter storm in Boston made American cancel my flight to London", ("US",)),
        ("Snowstorm in Minneapolis and my Delta flight to Detroit is delayed", ("US",)),
        ("Leaving JFK for Rome tomorrow, my flight is delayed", ("US",)),
        ("my flight from the United States to Paris was cancelled", ("US",)),
        ("I booked Manchester, England to Newark on United, the flight was late", ("US", "UK")),
        ("My flight from Timisoara, Romania to Chicago was delayed", ("US", "EU")),
        ("my domestic United flight was delayed 4 hours", ("US",)),
    ],
)
def test_stated_routes_do_not_ask(question, governing):
    route = detect_jurisdictions(question)
    assert route.needs_clarification is False
    assert route.governing == governing


@pytest.mark.parametrize(
    "question",
    [
        "How is compensation for a long flight delay different in the US versus the EU and UK?",
        "What's the phone number of Italy's EU261 enforcement body?",
        "Does Southwest fly to London?",
        "How much compensation for denied boarding?",
        "Can you book me on the next flight to Chicago?",   # personal, but nothing went wrong
    ],
)
def test_general_questions_do_not_ask(question):
    assert detect_jurisdictions(question).needs_clarification is False


def test_everyday_words_are_not_places():
    route = detect_jurisdictions("the agent was nice but we had to split up, my flight was delayed")
    assert route.endpoints == ()
    assert "EU" not in route.scope


def test_live_flight_data_answers_the_clarification():
    route = detect_jurisdictions("my Paris flight was cancelled", dep_iata="CDG", arr_iata="JFK")
    assert route.needs_clarification is False
    assert route.governing == ("US", "EU")


@pytest.mark.parametrize(
    "question,reply,governing",
    [
        ("my manchester to newark flight delayed", "the UK one", ("US", "UK")),
        ("my manchester to newark flight delayed", "US", ("US",)),
        ("My flight from Nantes to Chicago was cancelled", "France", ("US", "EU")),
        ("my delta flight was delayed 5 hours", "LHR to JFK", ("US", "UK")),
        ("my delta flight was delayed 5 hours", "from Atlanta to Paris", ("US",)),
        ("my paris flight is delayed", "CDG and then Detroit", ("US", "EU")),
    ],
)
def test_reply_to_a_clarification_settles_the_route(question, reply, governing):
    route = detect_jurisdictions(f"{question} {reply}", reply_start=len(question) + 1)
    assert route.governing == governing


@pytest.mark.parametrize(
    "question,reply,sentence,governing",
    [
        ("my delta flight was delayed 5 hours", "no idea",
         "Assuming this is a flight within the US.", ("US",)),
        ("my manchester to newark flight delayed", "not sure",
         "Assuming you're flying from Manchester, UK to Newark.", ("US", "UK")),
        ("my Timisoara flight was delayed, what am I owed?", "It's in Romania",
         "Assuming you're flying from Timisoara, Romania.", ("EU",)),
        ("My flight to London was cancelled", "don't know",
         "Assuming you're flying to London from a US airport.", ("US",)),
        ("my paris flight is delayed", "don't know",
         "Assuming you're flying from Paris.", ("EU",)),   # destination not guessed
    ],
)
def test_unanswered_clarification_answers_on_a_stated_assumption(question, reply, sentence, governing):
    from flight_delay.jurisdiction import assume_route

    route = detect_jurisdictions(f"{question} {reply}", reply_start=len(question) + 1)
    assumed, said = assume_route(route)
    assert said == sentence
    assert assumed.governing == governing


def test_unknown_city_is_not_guessed_even_after_asking():
    """EU/UK would promise €600 that may not be owed; elsewhere would withhold it."""
    from flight_delay.jurisdiction import assume_route

    q = "My flight from Nantes to Chicago was cancelled"
    assumed, said = assume_route(detect_jurisdictions(f"{q} dunno", reply_start=len(q) + 1))
    assert said.startswith("I couldn't confirm which country Nantes is in")
    assert {"US", "EU", "UK"} <= set(assumed.scope)
    assert assumed.governing == ("US",)          # Chicago is a named US endpoint


def test_resolve_regimes_is_the_applicability_matrix():
    from flight_delay.jurisdiction import resolve_regimes

    # (origin, dest): (governing, touches_us)
    matrix = {
        ("US", "US"): (("US",), True),
        ("US", "EU"): (("US",), True), ("US", "UK"): (("US",), True),
        ("EU", "US"): (("US", "EU"), True), ("UK", "US"): (("US", "UK"), True),
        ("UK", "EU"): (("UK",), False), ("EU", "UK"): (("EU",), False),
        ("EU", "OTHER"): (("EU",), False), ("UK", "OTHER"): (("UK",), False),
        ("OTHER", "EU"): ((), False), ("OTHER", "UK"): ((), False),
        # A departure outside US/EU/UK is never settled under US DOT, even into the
        # US (user, 2026-09-22): US stays retrievable ("may apply"), not governing.
        ("OTHER", "US"): ((), True), ("US", "OTHER"): (("US",), True),
        ("OTHER", "OTHER"): ((), False),
    }
    for (origin, dest), (governing, touches_us) in matrix.items():
        scope, got, touches = resolve_regimes(origin, dest)
        assert (got, touches) == (governing, touches_us), (origin, dest)
        # Both ends known and neither is the US: US DOT is not even retrievable.
        assert ("US" in scope) == touches_us, (origin, dest)


def test_unknown_endpoint_puts_us_in_scope_never_in_governing():
    from flight_delay.jurisdiction import resolve_regimes

    for origin, dest, governing in (
        ("EU", None, ("EU",)), ("UK", None, ("UK",)), ("OTHER", None, ()),
        (None, "EU", ()), (None, "OTHER", ()), (None, None, ()),
    ):
        scope, got, touches = resolve_regimes(origin, dest)
        assert got == governing and touches is False, (origin, dest)
        assert "US" in scope, (origin, dest)
    # ...but a US endpoint settles it, whichever end is unknown
    assert resolve_regimes(None, "US")[1:] == (("US",), True)
    assert resolve_regimes("US", None)[1:] == (("US",), True)


@pytest.mark.parametrize(
    "code,region",
    [("JFK", "US"), ("LAX", "US"), ("ORD", "US"), ("ATL", "US"),
     ("CDG", "EU"), ("FRA", "EU"), ("AMS", "EU"), ("ZRH", "EU"), ("KEF", "EU"),
     ("LHR", "UK"), ("MAN", "UK"),
     ("DXB", "OTHER"), ("DEL", "OTHER"), ("NRT", "OTHER"), ("YYZ", "OTHER")],
)
def test_region_of_reads_airport_regions_json(code, region):
    import json

    from flight_delay.jurisdiction import region_of

    table = json.loads((ROOT / "data" / "airport_regions.json").read_text(encoding="utf-8"))["airports"]
    assert table[code] == region          # the canonical file says so...
    assert region_of(code) == region      # ...and production agrees
    assert region_of(f" {code.lower()} ") == region


def test_region_of_never_defaults_to_us():
    """User decision 2026-09-16: a code the table does not list is UNKNOWN (None),
    not OTHER - the table lists OTHER airports explicitly - and never US."""
    from flight_delay.jurisdiction import _AIRPORT_REGIONS, is_known_airport, region_of

    assert "XQX" not in _AIRPORT_REGIONS
    assert region_of("XQX") is None and not is_known_airport("XQX")
    assert region_of("DXB") == "OTHER" and is_known_airport("DXB")   # a listed OTHER airport
    assert region_of(None) is None and region_of("") is None


def test_unknown_departure_code_in_flight_data_is_asked_about_not_treated_as_other():
    unknown_dep = _flight("XQX", "JFK")
    pipe, retriever = _pipeline({"UA57": unknown_dep})
    a = pipe.run("My flight UA57 was cancelled, what am I owed?", conversation_id="c1")
    assert a.outcome == "clarified" and "can't place airport code XQX" in a.text
    assert retriever.calls == []


def test_unknown_arrival_code_keeps_us_dot_retrievable_but_not_governing():
    from flight_delay.pipeline import route_filters

    live = _flight("CDG", "XQX")
    flt, route = route_filters("my flight UA57 was cancelled", "UA57", live)
    assert route.unknown_airport_codes == ("XQX",)
    assert "EU" in flt["jurisdiction_governing"] and "US" not in flt["jurisdiction_governing"]
    assert "US" in flt["jurisdiction_scope"]

    listed_other = _flight("CDG", "DXB")        # a listed OTHER airport: US is not in scope
    flt, _ = route_filters("my flight UA57 was cancelled", "UA57", listed_other)
    assert "US" not in flt["jurisdiction_scope"]


def test_typed_airport_codes_are_all_in_the_region_table():
    """Codes recognised in passenger text take their region from the JSON; a
    code missing from it would silently stop being recognised."""
    from flight_delay.jurisdiction import _AIRPORT_REGIONS, TEXT_AIRPORT_CODES

    assert not TEXT_AIRPORT_CODES - set(_AIRPORT_REGIONS)


def test_uppercase_words_that_are_airport_codes_elsewhere_are_not_places():
    """CFR is Caen and TSA is Taipei Songshan in the JSON, but not in a question."""
    route = detect_jurisdictions("Under 14 CFR Part 260 and TSA rules, what is my refund if cancelled?")
    assert route.endpoints == () and route.origin_place is None
    assert "EU" not in route.scope


@pytest.mark.parametrize(
    "dep,arr,touches_us",
    [
        ("JFK", "LAX", True), ("JFK", "CDG", True), ("CDG", "JFK", True),
        ("LHR", "JFK", True), ("JFK", "LHR", True), ("CDG", "LHR", False),
        ("CDG", "DXB", False), ("DXB", "CDG", False), ("DXB", "DEL", False),
    ],
)
def test_touches_us_only_with_a_us_airport(dep, arr, touches_us):
    route = detect_jurisdictions("my flight was delayed", dep_iata=dep, arr_iata=arr)
    assert route.touches_us is touches_us
    assert ("US" in route.governing) is touches_us
    assert ("US" in route.scope) is touches_us


def test_rest_of_world_route_gets_no_us_rules():
    """Never infer US from 'not EU/UK'."""
    route = detect_jurisdictions("my flight was delayed", dep_iata="DXB", arr_iata="CDG")
    assert (route.origin_region, route.destination_region) == ("OTHER", "EU")
    assert route.governing == () and route.scope == ("EU",)
    route = detect_jurisdictions("My flight from Dubai to Paris was cancelled")
    assert (route.origin_region, route.destination_region) == ("OTHER", "EU")
    assert route.governing == () and route.needs_clarification is False
    route = detect_jurisdictions("My flight from Toronto to Chicago was cancelled")
    assert (route.origin_region, route.destination_region) == ("OTHER", "US")
    # Into the US from outside US/EU/UK: US DOT MAY apply - retrievable, never governing
    # (user, 2026-09-22); pipeline.route_law_note tells the model to say "may".
    assert route.governing == () and route.scope == ("US",)


def test_named_us_airport_with_no_direction_touches_the_us():
    """Midway is one end of the flight, whichever end: US DOT governs (dom-34)."""
    route = detect_jurisdictions(
        "Our Southwest flight sat on the ground at Chicago Midway for 2 hours after landing")
    assert route.origin_region is None and route.destination_region is None
    assert route.governing == ("US",) and route.touches_us is True
    # An EU place with no direction does not make EU261 govern: departure is unknown.
    route = detect_jurisdictions("my Paris flight was cancelled, what am I owed?")
    assert route.governing == ()


def test_domestic_and_assumed_us_routes_are_us_not_other():
    from flight_delay.jurisdiction import assume_route

    route = detect_jurisdictions("my domestic United flight was delayed 4 hours")
    assert (route.origin_region, route.destination_region) == ("US", "US")
    q = "My flight to London was cancelled"
    assumed, _ = assume_route(detect_jurisdictions(f"{q} don't know", reply_start=len(q) + 1))
    assert (assumed.origin_region, assumed.destination_region) == ("US", "UK")
    q = "my delta flight was delayed 5 hours"
    assumed, _ = assume_route(detect_jurisdictions(f"{q} no idea", reply_start=len(q) + 1))
    assert (assumed.origin_region, assumed.destination_region) == ("US", "US")


def test_named_us_airport_does_not_cancel_domestic():
    """A US place in a 'domestic' question is still domestic (golden dom-32)."""
    route = detect_jurisdictions(
        "We've been sitting on the tarmac at Newark on a United domestic flight for 2.5 hours")
    assert route.domestic is True and route.needs_clarification is False
    assert (route.origin_region, route.destination_region) == ("US", "US")
    assert detect_jurisdictions("my domestic flight in Canada was delayed").domestic is False


def test_country_in_the_departure_region_is_not_the_destination():
    """'Canada's rules... from Toronto' must not become 'flying from Toronto to Canada'."""
    from flight_delay.jurisdiction import assume_route

    q = ("How much compensation do Canada's Air Passenger Protection Regulations give for a "
         "6-hour delay on my United flight from Toronto?")
    route = detect_jurisdictions(q)
    assert route.endpoints == () and route.needs_clarification is True   # where was it going?
    _, said = assume_route(detect_jurisdictions(f"{q} no idea", reply_start=len(q) + 1))
    assert said == "Assuming you're flying from Toronto."


def test_empty_jurisdiction_scope_drops_all_regulations():
    """DXB -> DEL: no regime applies. An empty scope must not read as 'no filter'."""
    store = MemoryStore()
    emb = HashEmbedder(256)
    us_law = _chunk("us1", "delay compensation rules")
    airline = _chunk("aa1", "delay compensation rules", airline="AA", doc_type="contract")
    store.upsert([us_law, airline], emb.embed_documents([c.embed_text for c in (us_law, airline)]))
    hits = store.sparse_search("delay compensation", 10, {"jurisdiction_scope": []})
    assert {cid for cid, _ in hits} == {"aa1"}

    from flight_delay.pipeline import route_filters

    live = FlightStatus(
        flight_iata="UA1", airline_iata="UA", status="active", dep_iata="DXB", dep_time=None,
        dep_estimated=None, dep_delayed=None, arr_iata="DEL", arr_time=None,
        arr_estimated=None, arr_delayed=None, delayed=None,
    )
    flt, _ = route_filters("my flight UA1 was delayed", "UA1", live)
    assert flt["jurisdiction_scope"] == [] and flt["jurisdiction_governing"] == []


# ==========================================================================
# Clarification flow
# ==========================================================================


class _FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, *, top_k=None, candidates_k=None, flt=None, exclude_topics=(), lanes=(), min_airline=None):
        from flight_delay.retrieval import RetrievalResult

        self.calls.append((query, flt))
        return RetrievalResult(chunks=[], timings_ms={}, n_dense=0, n_sparse=0, n_fused=0)


class _FakeTool:
    def __init__(self, flights):
        self.flights = flights
        self.quota = FileQuotaCounter(str(ROOT / ".pytest_quota_unused.json"), 1000)

    def get_flight(self, flight_no):
        return self.flights.get(flight_no)


class _FakeStore:
    def __init__(self):
        self.msgs = []

    def history(self, conv_id, limit=6):
        return self.msgs[-limit:]

    def save_message(self, conv_id, role, content, citations=None):
        self.msgs.append((role, content))


def _pipeline(flights=None):
    from flight_delay.generation import EchoLLM
    from flight_delay.pipeline import RagPipeline

    settings = Settings(embedder="hash", reranker="none", confidence_gate_enabled=False)
    retriever = _FakeRetriever()
    pipe = RagPipeline(retriever, EchoLLM(), _FakeTool(flights or {}), settings, store=_FakeStore())
    return pipe, retriever


def _flight(dep, arr):
    return FlightStatus(
        flight_iata="UA57", airline_iata="UA", status="cancelled", dep_iata=dep,
        dep_time=None, dep_estimated=None, dep_delayed=None, arr_iata=arr,
        arr_time=None, arr_estimated=None, arr_delayed=None, delayed=None,
    )


def _numbered(text):
    return [line for line in text.splitlines() if re.match(r"^\d+\. ", line)]


def test_no_place_asks_for_flight_number_and_both_airports_together():
    from flight_delay.pipeline import CLARIFY_INTRO

    pipe, retriever = _pipeline()
    a = pipe.run("my delta flight was delayed 5 hours", conversation_id="c1")
    assert a.clarification is True and a.intent == "clarify"
    assert a.text.startswith(CLARIFY_INTRO)
    items = _numbered(a.text)
    assert len(items) == 2
    assert "flight number" in items[0]
    assert "fly from" in items[1] and "flying to" in items[1]
    assert retriever.calls == []        # nothing retrieved, nothing generated


def test_manchester_gets_its_own_question():
    pipe, _ = _pipeline()
    a = pipe.run("my manchester to newark flight delayed", conversation_id="c1")
    items = _numbered(a.text)
    assert len(items) == 2                               # Newark is known: no airport question
    assert "Manchester, UK (MAN)" in items[1] and "New Hampshire" in items[1]


def test_unknown_city_asks_for_the_country():
    pipe, _ = _pipeline()
    a = pipe.run("my Timisoara flight was delayed, what am I owed?", conversation_id="c1")
    items = _numbered(a.text)
    assert len(items) == 3
    assert any("Which country are you flying from?" in i and "Timisoara" in i for i in items)


def test_flight_number_reply_is_looked_up_and_decides_the_law():
    pipe, retriever = _pipeline({"UA57": _flight("CDG", "EWR")})
    pipe.run("My United Paris flight was cancelled. What am I owed?", conversation_id="c1")
    a = pipe.run("UA57", conversation_id="c1")

    assert a.clarification is False
    assert a.assumption is None                      # verified by flight data, nothing assumed
    assert a.live_data is not None and a.live_data.dep_iata == "CDG"
    query, flt = retriever.calls[-1]
    assert "Paris flight was cancelled" in query      # the original question is resumed
    assert flt["jurisdiction_governing"] == ["US", "EU"]


def test_clarification_is_asked_once_then_the_answer_states_its_assumption():
    pipe, retriever = _pipeline()
    pipe.run("my manchester to newark flight delayed", conversation_id="c1")
    a = pipe.run("sorry, no idea", conversation_id="c1")
    assert a.clarification is False
    assert a.text.startswith("Assuming you're flying from Manchester, UK to Newark.")
    assert retriever.calls[-1][1]["jurisdiction_governing"] == ["US", "UK"]


def test_flight_number_that_airlabs_cannot_find_says_so_before_assuming_a_route():
    """
    The passenger answered the clarification with a flight number, the lookup ran
    and AirLabs had no record. The answer used to open with a bare "Assuming this
    is a flight within the US.", which is indistinguishable from never having
    looked - the report that found this read it as a missing API call.
    """
    pipe, _ = _pipeline()                    # _FakeTool knows no flights: every lookup misses
    pipe.run("My american airlines flight was cancelled today. what compensation do I get?",
             conversation_id="c1")
    a = pipe.run("Flight number AA2762", conversation_id="c1")
    assert a.clarification is False
    assert a.text.startswith(
        "I couldn't find flight AA2762 in the live flight data, so I couldn't confirm its "
        "route or status and the answer below isn't based on them.")
    assert "Assuming this is a flight within the US." in a.text   # and the assumption still shows


def test_a_flight_the_lookup_finds_is_named_instead_of_reported_missing():
    pipe, _ = _pipeline({"UA57": _flight("LHR", "ORD")})
    a = pipe.run("Is UA57 delayed? What am I owed?", conversation_id="c1")
    assert a.text.startswith("I found UA57: LHR to ORD, cancelled.")
    assert "couldn't find flight" not in a.text


def test_an_unsupported_carriers_flight_is_never_reported_as_not_found():
    """No lookup was made for BA117, so "I couldn't find it" would be a lie."""
    plan, calls = _plan("My BA117 from Heathrow to JFK was cancelled, what am I owed?")
    assert calls == [] and plan.flight_note is None


def test_reply_repeating_the_ambiguous_city_keeps_the_city_in_the_assumption():
    """Golden amb-06: 'The Manchester in England' qualified the city, not replaced it."""
    pipe, retriever = _pipeline()
    pipe.run("My Delta flight from Manchester to Atlanta was cancelled. What am I owed?", conversation_id="c1")
    a = pipe.run("The Manchester in England. I don't have the flight number.", conversation_id="c1")
    assert a.text.startswith("Assuming you're flying from Manchester, England to Atlanta.")
    assert retriever.calls[-1][1]["jurisdiction_governing"] == ["US", "UK"]


def test_country_only_reply_is_stated_as_the_assumption():
    pipe, retriever = _pipeline()
    pipe.run("my Timisoara flight was delayed, what am I owed?", conversation_id="c1")
    a = pipe.run("It's in Romania, I don't have the flight number", conversation_id="c1")
    assert a.text.startswith("Assuming you're flying from Timisoara, Romania.")
    flt = retriever.calls[-1][1]
    assert flt["jurisdiction_governing"] == ["EU"]       # destination still unknown
    assert "US" in flt["jurisdiction_scope"]             # but US refund rules stay retrievable


def test_non_us_departure_without_destination_asks_where_it_was_going():
    pipe, _ = _pipeline()
    a = pipe.run("My United flight from Paris was cancelled, what am I owed?", conversation_id="c1")
    assert a.clarification is True
    assert any("Which airport were you flying to?" in i for i in _numbered(a.text))


def test_flight_not_found_is_said_and_not_asked_again():
    pipe, _ = _pipeline()
    pipe.run("my delta flight was delayed 5 hours", conversation_id="c1")
    a = pipe.run("DL123", conversation_id="c1")
    assert a.clarification is False                       # asked once, not again
    assert a.text.startswith("I couldn't find flight DL123 in the live flight data")
    assert "Assuming this is a flight within the US." in a.text


def _golden_rows():
    import json

    return [json.loads(line) for line in
            (ROOT / "evals" / "golden_set.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_golden_builder_replays_production_routing():
    """The eval set's clarifications, assumptions and governing law come from
    pipeline.plan_turn(), so the two cannot drift apart."""
    from build_golden_set import check_route, expected_governing

    assert expected_governing("UK_to_EU") == ("UK",)
    assert expected_governing("US_to_EU") == ("US",)
    spec = {"route": "EU_to_US", "origin": None, "dest": None, "fixture": None, "reply": None,
            "question": "My flight to Paris was cancelled"}
    problems, first, final, reply = check_route(spec)
    assert first.clarification is not None
    assert any("implies governing" in p for p in problems)   # assumed US departure, tag says EU
    spec["question"] = "My flight from Paris to New York was cancelled"
    problems, first, _, _ = check_route(spec)
    assert problems == [] and first.clarification is None


def test_golden_builder_requires_a_fifth_edge_cases():
    """Fewer than 20% edge/ambiguous cases, overall or among answerable ones, fails the build."""
    from build_golden_set import edge_share_errors

    def rec(edge, answerable=True):
        return {"answerable": answerable, "tags": list(edge)}

    assert edge_share_errors([rec(["multi_leg"])] + [rec([])] * 4) == []   # exactly 20%
    assert len(edge_share_errors([rec(["multi_leg"])] + [rec([])] * 5)) == 2
    # traps count overall, but the answerable share is checked on its own
    errors = edge_share_errors([rec(["out_of_scope"], answerable=False)] * 2 + [rec([])] * 6)
    assert len(errors) == 1 and "answerable" in errors[0]


def test_golden_set_is_exactly_the_canonical_50():
    import json

    from golden.spec import CANONICAL_CASES, GOLDEN_SIZE

    rows = [json.loads(line) for line in
            (ROOT / "evals" / "golden_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    assert GOLDEN_SIZE == 50 and len(ids) == 50 and len(set(ids)) == 50
    assert ids == list(CANONICAL_CASES)
    assert len({r["question"].lower() for r in rows}) == 50


def test_golden_record_behaviour_answerability_and_clarification_agree():
    """Every record is exactly one of three shapes, so no case can both promise an
    answer and carry a clarifying question (or claim to clarify while marked
    answerable). answer -> no clarification; clarify/abstain -> not answerable."""
    import json

    rows = [json.loads(line) for line in
            (ROOT / "evals" / "golden_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    shapes = {("answer", True, False), ("clarify", False, True), ("abstain", False, False)}
    for r in rows:
        shape = (r["expected_behavior"]["action"], r["answerable"],
                 r["expected_clarification"] is not None)
        assert shape in shapes, f"{r['id']}: {shape}"
        if r["expected_behavior"]["action"] == "clarify":
            # expected_output IS the question asked, never a drafted answer.
            assert r["expected_output"] == r["expected_clarification"], r["id"]
        else:
            # No "Assuming you're flying from ..." opener on a record that says
            # the question needs no clarification.
            assert not r["expected_output"].startswith("Assuming"), r["id"]


def test_golden_builder_rejects_a_set_that_is_not_the_canonical_50():
    from build_golden_set import canonical_errors, load_cases

    cases = load_cases()
    assert canonical_errors(cases) == []
    assert any("no spec" in e for e in canonical_errors(cases[1:]))
    extra = {**cases[0], "id": "dom-99"}
    assert any("not in spec.CANONICAL_CASES" in e for e in canonical_errors([*cases, extra]))
    assert any("more than once" in e for e in canonical_errors([*cases, cases[0]]))


def test_golden_set_covers_what_issues_md_requires():
    import json

    from golden.spec import DIFFICULTIES, EDGE_KINDS

    rows = [json.loads(line) for line in
            (ROOT / "evals" / "golden_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    scenarios = [r["scenario"] for r in rows]
    carriers = {s["airline"] for s in scenarios}
    assert {"AA", "DL", "UA", "WN", "BA"} <= carriers
    jurisdictions = {s["jurisdiction"] for s in scenarios}
    assert any("US" in j for j in jurisdictions)
    assert any("EU" in j for j in jurisdictions)
    assert any("UK" in j for j in jurisdictions)
    kinds = {s["disruption_type"] for s in scenarios}
    assert {"delay", "cancellation", "denied_boarding", "missed_connection", "tarmac_delay"} <= kinds
    edges = {t for r in rows for t in r["tags"] if t in EDGE_KINDS}
    assert {"ambiguous_route", "unsupported_airline", "us_territory", "other_region"} <= edges
    actions = {r["expected_behavior"]["action"] for r in rows}
    assert actions == {"answer", "clarify", "abstain"}
    assert 5 <= sum(r["category"] == "hybrid_live_data" for r in rows) <= 7
    difficulties = {t for r in rows for t in r["tags"] if t in DIFFICULTIES}
    assert difficulties == {"easy", "medium", "hard"}


def test_eval_run_gives_hybrid_cases_their_flight_data_without_airlabs():
    """Both evaluation stages replay the golden fixture instead of the real
    AirLabs tool, so the synthetic 99xx flights get their data and a run with
    AIRLABS_KEY set spends no live calls."""
    from golden.replay import run_conversation

    class _NoAirLabs:
        quota = FileQuotaCounter(str(ROOT / ".pytest_quota_unused.json"), 1000)

        def get_flight(self, flight_no):
            raise AssertionError(f"eval called the real flight tool for {flight_no}")

    rows = _golden_rows()
    hybrids = [r for r in rows if r.get("flight_fixture")]
    assert len(hybrids) >= 5
    pipe, _ = _pipeline()
    real_tool = pipe.tool = _NoAirLabs()
    for item in hybrids:
        ans = run_conversation(pipe, item)
        assert ans.live_data is not None, item["id"]
        assert ans.live_data.flight_iata == item["flight_fixture"]["flight_iata"]
        assert ans.live_data.dep_iata == item["flight_fixture"]["dep_iata"]
    assert pipe.tool is real_tool                      # restored after each case
    # a non-hybrid case that names a flight gets no data rather than a live lookup
    ans = run_conversation(pipe, {"id": "x", "question": "Is AA2359 delayed?"})
    assert ans.live_data is None


def test_us_rules_apply_whenever_the_route_touches_the_us():
    """DOT rules ride along on any itinerary touching the US, in either
    direction, and on domestic ones."""
    for q in (
        "my flight from Paris to New York was delayed",
        "my flight from New York to Paris was delayed",
        "my flight from Atlanta to Dallas was delayed",
    ):
        assert "US" in detect_jurisdictions(q).governing, q
    # No route at all: US DOT is retrievable, not governing.
    route = detect_jurisdictions("how much for denied boarding?")
    assert route.scope == ("US",) and route.governing == ()


def test_live_flight_data_overrides_the_question_text():
    """The question says Paris; the actual flight departed JFK. Real airport
    codes win, because passengers describe trips loosely."""
    route = detect_jurisdictions("my Paris flight", dep_iata="JFK", arr_iata="CDG")
    assert route.governing == ("US",)


def test_lowercase_words_are_not_treated_as_airport_codes():
    """'man' is a word; MAN is Manchester. Only uppercase tokens are codes."""
    assert detect_jurisdictions("the man at the gate said no").scope == ("US",)


def test_united_kingdom_is_not_united_airlines():
    """Regression: the airline pattern matched the 'United' in 'United Kingdom',
    scoping a UK question to United's documents."""
    from flight_delay.pipeline import detect_airline

    assert detect_airline("My flight from the United Kingdom was cancelled") is None
    assert detect_airline("My United flight was cancelled") == "UA"


def test_jurisdiction_scope_keeps_airline_docs_but_filters_regulations():
    """Airline contracts are filed under US yet govern the carrier everywhere,
    so a Europe question must keep them while narrowing which law applies."""
    store = MemoryStore()
    emb = HashEmbedder(256)
    us_law = _chunk("us1", "denied boarding compensation rules")
    eu_law = Chunk(
        chunk_id="eu1", doc_id="eu", doc_title="EU261", publisher="EU",
        jurisdiction="EU", section_id="7", breadcrumb="Article 7",
        text="denied boarding compensation rules", embed_text="denied boarding compensation rules",
        source_url="u", doc_type="regulation", token_estimate=10,
    )
    airline = _chunk("aa1", "denied boarding compensation rules",
                     airline="AA", doc_type="contract")
    store.upsert([us_law, eu_law, airline],
                 emb.embed_documents([c.embed_text for c in (us_law, eu_law, airline)]))

    hits = store.sparse_search("denied boarding", 10, {"jurisdiction_scope": ["US", "EU"]})
    assert {cid for cid, _ in hits} == {"us1", "eu1", "aa1"}

    hits = store.sparse_search("denied boarding", 10, {"jurisdiction_scope": ["US"]})
    ids = {cid for cid, _ in hits}
    assert "eu1" not in ids       # EU law dropped for a purely domestic question
    assert "aa1" in ids           # airline contract survives
    assert "us1" in ids


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Is UA2402 late", "UA2402"),
        ("flight BA 117", "BA117"),
        ("What changed in 2024", None),   # 'in' is lowercase -> not a flight
        ("I paid 400 dollars", None),
        ("no flight here", None),
        # shaped like flight numbers, but are regulations, aircraft or airport places
        ("Does UK261 apply to my flight?", None),
        ("what does EU 261 say", None),
        ("Which seats on Delta's A350 have legroom?", None),
        ("my Delta flight at gate B12 was delayed", None),
        ("Does UK261 apply to BA117?", "BA117"),
    ],
)
def test_flight_number_extraction(text, expected):
    assert extract_flight_number(text) == expected


# ==========================================================================
# US territories
# ==========================================================================


@pytest.mark.parametrize("code", ["SJU", "BQN", "STT", "STX", "GUM", "SPN", "PPG"])
def test_us_territory_airports_are_us(code):
    """14 CFR 250.9: 'points within the United States (including the territories
    and possessions)'. The JSON is the only table that says so."""
    import json

    from flight_delay.jurisdiction import region_of

    table = json.loads((ROOT / "data" / "airport_regions.json").read_text(encoding="utf-8"))["airports"]
    assert table[code] == "US"
    assert region_of(code) == "US"


def test_territory_regulation_text_is_in_the_corpus():
    """The reclassification rests on this sentence of the Part 250 notice."""
    from flight_delay.ingest import parse_document

    pdf = ROOT / "data" / "government_mandates" / "US" / "14_CFR_Part250_oversales.pdf"
    sections = parse_document(pdf, doc_id="us-cfr-250-oversales", doc_title="14 CFR Part 250",
                              publisher="DOT", jurisdiction="US", doc_type="regulation")
    text = " ".join(" ".join(s.text.split()) for s in sections)
    assert "within the United States (including the territories and possessions)" in text


def test_territory_routes_are_us_dot_routes():
    """Guam -> Saipan (both territories) used to be OTHER -> OTHER: no US DOT at all."""
    route = detect_jurisdictions("my flight was cancelled", dep_iata="GUM", arr_iata="SPN")
    assert (route.origin_region, route.destination_region) == ("US", "US")
    assert route.governing == ("US",) and route.touches_us is True
    # San Juan -> London: a US departure, so UK261 does not govern
    route = detect_jurisdictions("my flight was delayed", dep_iata="SJU", arr_iata="LHR")
    assert route.governing == ("US",)
    # ...and London -> San Juan is UK261 + US DOT, like any UK -> US flight
    route = detect_jurisdictions("my flight was delayed", dep_iata="LHR", arr_iata="SJU")
    assert route.governing == ("US", "UK")
    # typed codes take the same region from the JSON
    route = detect_jurisdictions("my flight from STT to SJU was cancelled")
    assert (route.origin_region, route.destination_region) == ("US", "US")
    assert route.needs_clarification is False


def test_region_counts_in_airport_regions_meta_match_the_table():
    import json
    from collections import Counter

    data = json.loads((ROOT / "data" / "airport_regions.json").read_text(encoding="utf-8"))
    assert dict(Counter(data["airports"].values())) == data["_meta"]["regions"]


# ==========================================================================
# Unsupported airlines
# ==========================================================================


def _plan(question, history=(), fetched=None):
    """plan_turn with a flight lookup that records every flight number it is asked for."""
    from flight_delay.pipeline import plan_turn

    calls = []

    def fetch(flight_no):
        calls.append(flight_no)
        return (fetched or {}).get(flight_no)

    return plan_turn(question, list(history), fetch), calls


def test_supported_airlines_is_the_single_definition():
    from flight_delay import pipeline, tools

    assert sorted(tools.SUPPORTED_AIRLINES) == ["AA", "DL", "UA", "WN"]
    assert pipeline.SUPPORTED_AIRLINES is tools.SUPPORTED_AIRLINES


@pytest.mark.parametrize(
    "question,flight,carrier,supported",
    [
        ("Is BA117 delayed?", "BA117", "BA", False),
        ("Is AF 22 on time?", "AF22", "AF", False),
        ("Is AA2359 delayed?", "AA2359", "AA", True),
        ("my Delta flight at gate B12 was late", None, "DL", True),
    ],
)
def test_flight_number_and_supported_airline_are_separate_questions(question, flight, carrier, supported):
    from flight_delay.pipeline import detect_airline, resolve_airline
    from flight_delay.tools import SUPPORTED_AIRLINES

    flight_no = extract_flight_number(question)
    assert flight_no == flight                        # recognised as a flight number...
    assert resolve_airline(question, flight_no) == carrier
    assert (carrier in SUPPORTED_AIRLINES) is supported   # ...supported or not
    assert detect_airline(question, flight_no) == (carrier if supported else None)


def test_ba117_is_an_unsupported_british_airways_flight():
    plan, calls = _plan("Is BA117 delayed today? Do I get anything if it is?")
    assert plan.flight_no == "BA117" and plan.unsupported_airline == "BA"
    assert calls == []                                   # no AirLabs lookup
    assert plan.live is None and plan.clarification is None
    assert "BA117 is a flight on British Airways" in plan.unsupported_reply
    assert "American Airlines, Delta, United and Southwest" in plan.unsupported_reply


def test_another_unsupported_prefix_is_declined_the_same_way():
    plan, calls = _plan("My AF 22 was cancelled, what am I owed?")
    assert plan.unsupported_airline == "AF" and calls == []
    assert plan.unsupported_reply is not None and "Air France" in plan.unsupported_reply


def test_unsupported_airline_by_name_is_not_answered_from_supported_airlines_policies():
    plan, calls = _plan("How much compensation will JetBlue pay me for a 5-hour delay?")
    assert plan.unsupported_airline == "B6" and calls == []
    assert plan.unsupported_reply is not None
    assert plan.flt["airline_scope"] == "B6"


def test_supported_flight_is_looked_up_and_scoped_to_its_airline():
    live = FlightStatus(
        flight_iata="AA2359", airline_iata="AA", status="landed", dep_iata="DEN", dep_time=None,
        dep_estimated=None, dep_delayed=114, arr_iata="DFW", arr_time=None,
        arr_estimated=None, arr_delayed=98, delayed=98,
    )
    plan, calls = _plan("My flight AA2359 was delayed, what am I owed?", fetched={"AA2359": live})
    assert calls == ["AA2359"]
    assert plan.unsupported_airline is None and plan.unsupported_reply is None and plan.notice is None
    assert plan.live is live and plan.flt["airline_scope"] == "AA"
    assert plan.flt["jurisdiction_governing"] == ["US"]


def test_unsupported_airline_with_a_known_route_gets_government_rules_only():
    plan, calls = _plan("My BA117 from London Heathrow to New York JFK was cancelled. What does UK261 give me?")
    assert calls == [] and plan.unsupported_reply is None
    assert plan.notice.startswith("I can only help with flights on American Airlines")
    assert "only government passenger-rights rules" in plan.notice
    assert plan.flt["airline_scope"] == "BA"              # no AA/DL/UA/WN documents
    assert plan.flt["jurisdiction_governing"] == ["US", "UK"]


def test_unsupported_airline_general_legal_question_is_answered_from_regulations():
    plan, calls = _plan("Under UK261, what compensation is owed when a flight like BA117 is cancelled?")
    assert calls == [] and plan.unsupported_reply is None and plan.notice is not None
    assert plan.flt["airline_scope"] == "BA"


def test_unsupported_carrier_arriving_in_europe_flags_the_carrier_dependent_rule():
    plan, _ = _plan("My BA178 from New York JFK to London Heathrow was delayed 5 hours. What are my rights?")
    assert plan.flt["jurisdiction_governing"] == ["US"]
    assert "Whether UK261 also covers a flight arriving in the UK" in plan.notice


def test_unsupported_airline_retrieves_no_supported_airline_contract():
    """End to end: the retrieval filter drops every airline document, and the answer opens
    with the notice."""
    store = MemoryStore()
    emb = HashEmbedder(256)
    chunks = [
        _chunk("uk261", "cancellation compensation refund re-routing care"),
        _chunk("aa", "cancellation compensation refund policy", airline="AA", doc_type="contract"),
        _chunk("dl", "cancellation compensation refund policy", airline="DL", doc_type="policy"),
        _chunk("ua", "cancellation compensation refund policy", airline="UA", doc_type="service_plan"),
        _chunk("wn", "cancellation compensation refund policy", airline="WN", doc_type="contract"),
    ]
    store.upsert(chunks, emb.embed_documents([c.embed_text for c in chunks]))
    plan, _ = _plan("My BA117 from London Heathrow to New York JFK was cancelled. What compensation or refund?")
    hits = store.sparse_search("cancellation compensation refund", 10,
                               {**plan.flt, "jurisdiction_scope": ["US"]})
    assert {cid for cid, _ in hits} == {"uk261"}

    pipe, retriever = _pipeline()
    a = pipe.run("My BA117 from London Heathrow to New York JFK was cancelled. What does UK261 give me?")
    assert a.unsupported_airline == "BA" and a.live_data is None
    assert a.text.startswith("I can only help with flights on American Airlines")
    assert retriever.calls and all(flt["airline_scope"] == "BA" for _, flt in retriever.calls)


def test_unsupported_flight_never_reaches_airlabs(airlabs):
    tool = airlabs(use_flight_fixtures=False)
    assert tool.get_flight("BA117") is None
    assert _FakeHttp.calls == []
    pipe, retriever = _pipeline()
    pipe.tool = tool
    a = pipe.run("Is BA117 delayed?")
    assert a.intent == "unsupported" and _FakeHttp.calls == [] and retriever.calls == []


# ==========================================================================
# Quota guard
# ==========================================================================


class _FakeHttp:
    """Stands in for httpx.Client, so no test ever reaches AirLabs."""

    calls: list = []

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None):
        _FakeHttp.calls.append((url, params))

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"response": {"flight_iata": "AA2359", "airline_iata": "AA",
                                     "dep_iata": "JFK", "arr_iata": "LAX", "status": "live"}}
        return R()


@pytest.fixture
def airlabs(tmp_path, monkeypatch):
    from flight_delay import tools

    _FakeHttp.calls = []
    monkeypatch.setattr(tools.httpx, "Client", _FakeHttp)

    def make(**settings):
        s = Settings(airlabs_key="test-key", **settings)
        return tools.AirLabsTool(s, fixtures_dir=str(ROOT / "data" / "fixtures" / "flights"),
                                 quota_path=str(tmp_path / "quota.json"))
    return make


def test_fixtures_are_off_by_default(monkeypatch):
    monkeypatch.delenv("USE_FLIGHT_FIXTURES", raising=False)
    assert Settings(_env_file=None).use_flight_fixtures is False


def test_fixture_mode_on_serves_the_recording_without_http(airlabs):
    assert (ROOT / "data" / "fixtures" / "flights" / "AA2359.json").exists()
    fs = airlabs(use_flight_fixtures=True).get_flight("AA2359")
    assert fs is not None and fs.dep_iata == "DEN"      # the recorded snapshot
    assert _FakeHttp.calls == []


def test_fixture_mode_off_goes_to_the_live_api(airlabs):
    fs = airlabs(use_flight_fixtures=False).get_flight("AA2359")
    assert len(_FakeHttp.calls) == 1                     # the (mocked) live call
    assert _FakeHttp.calls[0][1]["flight_iata"] == "AA2359"
    assert fs is not None and fs.dep_iata == "JFK"       # live data, not the fixture


def test_fixture_mode_off_respects_allow_live(airlabs):
    assert airlabs(use_flight_fixtures=False).get_flight("AA2359", allow_live=False) is None
    assert _FakeHttp.calls == []


# ==========================================================================
# System prompt
# ==========================================================================


def test_system_prompt_is_the_canonical_file():
    """One prompt: production's SYSTEM_PROMPT is system_prompt.md, not a copy."""
    from flight_delay import generation

    expected = (ROOT / "system_prompt.md").read_text(encoding="utf-8").strip()
    assert generation.SYSTEM_PROMPT
    assert expected == generation.SYSTEM_PROMPT
    assert generation.SYSTEM_PROMPT_PATH == ROOT / "system_prompt.md"


def test_missing_system_prompt_fails_loudly(tmp_path):
    from flight_delay.generation import load_system_prompt

    with pytest.raises(RuntimeError, match="System prompt not found"):
        load_system_prompt(tmp_path / "system_prompt.md")


def test_no_prompt_copy_lives_in_python_code():
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "JURISDICTIONAL HIERARCHY" not in text and "NEVER ASSERT DELAY CAUSE" not in text, path


def test_system_prompt_carries_the_restored_rules():
    """Rules restored/tightened on 2026-09-14 (issues.md #3)."""
    from flight_delay.generation import SYSTEM_PROMPT as p

    flat = " ".join(p.split())
    assert "LATER effective date" in flat                          # conflicting sources
    assert "NOT a legal or policy source" in flat                  # flight data
    assert "Direction unknown" in flat and "do NOT pick a direction" in flat
    assert "End with this caveat ONLY when" in flat                # cause caveat scoped
    assert "Never supply an amount, band or cap from memory" in flat
    assert "€600" not in p and "£520" not in p                     # no amounts to parrot


# ==========================================================================
# API: /ask is the one production answer path
# ==========================================================================


class _ScriptedLLM:
    """Returns the scripted answers in order, so a validation retry is observable."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return self.answers.pop(0)

    def stream(self, system, user):  # pragma: no cover - must never be used by the API
        raise AssertionError("token streaming bypasses validation")


class _OneChunkRetriever(_FakeRetriever):
    def retrieve(self, query, *, top_k=None, candidates_k=None, flt=None, exclude_topics=(), lanes=(), min_airline=None):
        from flight_delay.retrieval import RetrievalResult

        self.calls.append((query, flt))
        ch = _chunk("p260", "A refund is owed when an airline cancels a flight and the passenger "
                            "chooses not to travel.")
        return RetrievalResult(chunks=[ch], timings_ms={}, n_dense=1, n_sparse=1, n_fused=1)


@pytest.fixture
def api_client(monkeypatch):
    from fastapi.testclient import TestClient

    from flight_delay import api

    api._BUCKET.clear()

    def make(pipeline, **settings):
        monkeypatch.setitem(api.STATE, "pipeline", pipeline)
        monkeypatch.setitem(api.STATE, "settings", Settings(embedder="hash", reranker="none", **settings))
        return TestClient(api.app)   # no `with`: lifespan (Postgres, models) is not started
    return make


def test_ui_posts_to_ask_not_to_the_stream_endpoint():
    from flight_delay.api import HTML_PAGE

    assert "fetch('/ask'," in HTML_PAGE
    assert "/ask/stream" not in HTML_PAGE


def test_ui_hides_every_citation_marker_from_the_visible_answer():
    """Session 19: markers are for the validator; the passenger reads prose and gets
    the source list separately. The old code stripped only markers present in
    d.citations, so one the model invented, or one whose source was dropped from the
    list, survived on screen as "[S5]"."""
    import re

    from flight_delay.api import HTML_PAGE

    line = next(ln for ln in HTML_PAGE.splitlines() if "let shown=" in ln)
    # what the browser receives must be a real regex: no literal tab, no doubled
    # backslashes (HTML_PAGE is a plain, non-raw Python string)
    assert "\t" not in line and "\\\\" not in line
    pattern = re.search(r"replace\(/(.+?)/g,", line).group(1)

    strip = lambda s: re.sub(pattern, "", s)  # noqa: E731 - the browser's own pattern
    assert strip("You are entitled to a full cash refund [S1].") == \
        "You are entitled to a full cash refund."
    assert strip("compensation, and care such as meals [S5][S3].") == \
        "compensation, and care such as meals."
    assert strip("meals, refreshments and a hotel [S5, S8].") == \
        "meals, refreshments and a hotel."
    assert strip("a full refund [S5, S6][S1].") == "a full refund."
    assert strip("inform you [S12] and pay [S7] within seven days [S2].") == \
        "inform you and pay within seven days."
    # a marker with no matching citation is still hidden
    assert "[S" not in strip("Unsupported claim [S99] stays invisible.")
    assert strip("No markers at all here.") == "No markers at all here."


def test_flight_card_still_reports_a_missing_delay_as_none_reported():
    """Regression for the Session 18 UI fix: 'n/a' read as 'we don't know', when the
    record positively says no delay was reported."""
    from flight_delay.api import HTML_PAGE

    assert "'none reported'" in HTML_PAGE
    assert "delay cause:" in HTML_PAGE


def test_ask_applies_the_confidence_gate(api_client):
    from flight_delay.pipeline import RagPipeline

    llm = _ScriptedLLM()                          # generation must not be reached
    settings = Settings(embedder="hash", reranker="none", confidence_gate_enabled=True)
    pipe = RagPipeline(_FakeRetriever(), llm, _FakeTool({}), settings, store=_FakeStore())
    r = api_client(pipe).post("/ask", json={"question": "How much compensation for denied boarding?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] == [] and llm.prompts == []      # gated before generation


def test_ask_never_tells_the_passenger_about_rate_limits_or_quota(api_client):
    """User's decision, Session 19: a provider limit is an operational fact, reported
    INTERNALLY. The passenger gets one neutral sentence - they cannot act on "tokens
    per day", it invites them to retry into a wall, and it leaks how the service is
    provisioned. The full provider message still goes to the turn log and to
    diagnostics; AskResponse simply has no field that carries it."""
    import json as _json

    from flight_delay.generation import LLMTransientError
    from flight_delay.pipeline import RagPipeline

    provider_message = ("Rate limit reached for model `openai/gpt-oss-20b` in organization "
                        "`org_SECRET1` on tokens per day (TPD): Limit 200000, Used 196164. "
                        "Please try again in 48m35s.")

    class RateLimited:
        last = None
        max_tokens = 900

        def complete(self, system, user):  # noqa: ARG002
            raise LLMTransientError(f"groq asked to retry after 2915s: HTTP 429: {provider_message}",
                                    status=429, provider_code="rate_limit_exceeded",
                                    provider_message=provider_message, retry_after_s=2915.0)

    settings = Settings(embedder="hash", reranker="none", confidence_gate_enabled=False)
    pipe = RagPipeline(_OneChunkRetriever(), RateLimited(), _FakeTool({}), settings,
                       store=_FakeStore())
    body = api_client(pipe).post(
        "/ask", json={"question": "Can I get a refund if American cancels my domestic flight?"}).json()

    assert body["outcome"] == "llm_error"
    assert "answer service is unavailable" in body["answer"]
    # nothing anywhere in the response may hint at the provider or its limits
    # ...except the conversation token, a random HMAC that hits these digit runs by
    # chance about once in a hundred runs (it carries no provider information).
    blob = _json.dumps({k: v for k, v in body.items() if k != "conversation_token"}).lower()
    for leaked in ("tokens per day", "tpd", "rate limit", "quota", "429", "2915",
                   "org_secret1", "groq", "200000", "196164", "retry after"):
        assert leaked not in blob, f"{leaked!r} reached the passenger"


def test_ask_validates_and_retries_a_bad_answer(api_client):
    from flight_delay.pipeline import RagPipeline

    llm = _ScriptedLLM(
        "You must be refunded within 7 business days for a cancelled flight.",      # uncited
        "You are owed a refund if the airline cancels and you choose not to travel [S1].",
    )
    settings = Settings(embedder="hash", reranker="none", confidence_gate_enabled=False)
    pipe = RagPipeline(_OneChunkRetriever(), llm, _FakeTool({}), settings, store=_FakeStore())
    body = api_client(pipe).post(
        "/ask", json={"question": "Can I get a refund if American cancels my domestic flight?"}).json()
    assert body["retry_count"] == 1 and len(llm.prompts) == 2
    assert "YOUR PREVIOUS ANSWER WAS REJECTED" in llm.prompts[1]
    # and it must quote the sentence that was rejected, not only count it
    assert "You must be refunded within 7 business days" in llm.prompts[1]
    assert body["validation_failures"] == []
    assert [c["marker"] for c in body["citations"]] == ["S1"]


def test_stream_endpoint_is_not_exposed_by_default(api_client):
    pipe, _ = _pipeline()
    client = api_client(pipe)
    assert client.post("/ask/stream", json={"question": "How much for denied boarding?"}).status_code == 404
    assert "/ask/stream" not in client.get("/openapi.json").json()["paths"]
    assert "/ask" in client.get("/openapi.json").json()["paths"]


def test_stream_endpoint_when_enabled_replays_the_validated_ask_answer(api_client):
    """No second pipeline: the internal stream runs pipeline.run(), never llm.stream()."""
    from flight_delay.pipeline import RagPipeline

    llm = _ScriptedLLM(
        "You must be refunded within 7 business days for a cancelled flight.",
        "You are owed a refund if the airline cancels and you choose not to travel [S1].",
    )
    settings = Settings(embedder="hash", reranker="none", confidence_gate_enabled=False)
    pipe = RagPipeline(_OneChunkRetriever(), llm, _FakeTool({}), settings, store=_FakeStore())
    r = api_client(pipe, enable_stream_endpoint=True).post(
        "/ask/stream", json={"question": "Can I get a refund if American cancels my domestic flight?"})
    assert r.status_code == 200
    events = re.findall(r"event: (\w+)\ndata: (.*)\n\n", r.text)
    assert [e for e, _ in events] == ["meta", "token", "validation", "done"]
    import json

    validation = json.loads(events[2][1])
    assert validation["ok"] is True and validation["retry_count"] == 1
    assert "[S1]" in json.loads(events[1][1])["t"]


def test_quota_counter_persists_and_increments(tmp_path):
    """100 calls per MONTH means one runaway loop destroys the month. The guard
    is the difference between a degraded demo and a dead API key."""
    p = tmp_path / "q.json"
    c = FileQuotaCounter(str(p), monthly_quota=100)
    assert c.remaining() == 100
    for _ in range(5):
        c.increment()
    assert FileQuotaCounter(str(p), monthly_quota=100).remaining() == 95


# ==========================================================================
# Stage 1: retrieval evaluation (evals/retrieval_eval.py)
# ==========================================================================


def test_retrieval_metrics_stay_within_bounds_under_fuzzing():
    """Regression + fuzz for the nDCG > 1.0 bug caused by duplicate sections, and
    a bounds check on every other rate the table prints."""
    import retrieval_eval as rev

    random.seed(1234)
    for _ in range(1000):
        retrieved = [random.choice("abcde") for _ in range(random.randint(0, 15))]
        gold = set(random.sample("abcde", random.randint(0, 3)))
        assert 0.0 <= rev.ndcg_at_k(retrieved, gold, 5) <= 1.0
        assert 0.0 <= rev.section_recall_at_k(retrieved, gold, 3) <= 1.0
        assert 0.0 <= rev.mrr(retrieved, gold) <= 1.0

    for _ in range(200):
        gold = [" ".join(random.choice("abcdefgh") for _ in range(random.randint(1, 20)))
                for _ in range(random.randint(0, 3))]
        ctx = [" ".join(random.choice("abcdefgh") for _ in range(random.randint(1, 30)))
               for _ in range(random.randint(0, 4))]
        m = rev.evidence_metrics(gold, ctx)
        assert all(0.0 <= v <= 1.0 for v in m.values()), m


def test_retrieval_metrics_reward_higher_rank():
    import retrieval_eval as rev

    assert rev.ndcg_at_k(["a", "b"], {"a"}, 5) > rev.ndcg_at_k(["b", "a"], {"a"}, 5)
    assert rev.mrr(["x", "a"], {"a"}) == 0.5
    # section_recall@3 is a CUTOFF: a gold section at rank 4 does not count.
    assert rev.section_recall_at_k(["x", "y", "a"], {"a"}, 3) == 1.0
    assert rev.section_recall_at_k(["x", "y", "z", "a"], {"a"}, 3) == 0.0


def test_section_metrics_use_doc_qualified_section_ids():
    """EU261 and UK261 both have 'article-7-right-to-compensation'. Scored on the
    bare section id, one regulation counts as a hit for the other - which is the
    difference between telling a passenger they are owed EUR600 and nothing."""
    import retrieval_eval as rev

    gold = {"eu261#article-7-right-to-compensation"}
    wrong_regulation = ["uk261#article-7-right-to-compensation"]
    right_regulation = ["eu261#article-7-right-to-compensation"]
    assert rev.section_recall_at_k(wrong_regulation, gold, 3) == 0.0
    assert rev.mrr(wrong_regulation, gold) == 0.0
    assert rev.ndcg_at_k(wrong_regulation, gold, 5) == 0.0
    assert rev.section_recall_at_k(right_regulation, gold, 3) == 1.0

    # and the metric the sweep actually feeds is built from doc_id#section_id
    rows = _golden_rows()
    assert all("#" in s for r in rows for s in r["gold_doc_sections"])


def test_recall_at_5_is_absent_everywhere():
    """section_recall@3 is the only rank-cutoff recall metric. Recall@5 must not
    reappear in code, output, schemas, the Makefile or active documentation."""
    import re as _re

    pattern = _re.compile(r"recall(@|_at_)(?!3\b)\d", _re.I)
    # README.md is deliberately not an input to any test: it is an optional overview.
    # The alert rules are included because a dead recall@20 alert survived there.
    files = [
        ROOT / "evals" / "retrieval_eval.py", ROOT / "evals" / "generation_eval.py",
        ROOT / "evals" / "build_golden_set.py", ROOT / "Makefile",
        ROOT / "RUNBOOK.md", ROOT / "RESULTS.md", ROOT / "CLAUDE.md",
        ROOT / "deploy" / "prometheus" / "rules" / "rag.yml",
    ]
    for f in files:
        if f in LOCAL_ONLY_DOCS and not f.exists():
            continue
        hits = pattern.findall(f.read_text(encoding="utf-8"))
        assert not hits, f"{f.name} still mentions {hits}"


def test_chunk_selection_rule_is_the_documented_one():
    """evidence_recall within 0.02 of best -> highest MRR -> lowest context_tokens
    -> smaller chunk. Checked on synthetic rows, one tie-break at a time."""
    from retrieval_eval import select_configuration

    def row(size, recall, mrr_, tokens, overlap=15):
        return {"chunk_target_tokens": size, "chunk_overlap_pct": overlap,
                "evidence_recall": recall, "mrr": mrr_, "context_tokens": tokens}

    # 1-2: clearly best recall wins even with a worse MRR
    pick = select_configuration([row(256, 0.50, 0.9, 900), row(550, 0.80, 0.4, 2000)])
    assert pick["chunk_target_tokens"] == 550
    # 3: inside the 0.02 band, MRR decides
    pick = select_configuration([row(256, 0.79, 0.9, 2000), row(550, 0.80, 0.4, 900)])
    assert pick["chunk_target_tokens"] == 256
    # 4: tied on recall and MRR, fewest context tokens decides
    pick = select_configuration([row(1024, 0.80, 0.5, 4000), row(256, 0.80, 0.5, 900)])
    assert pick["chunk_target_tokens"] == 256
    # 5: tied on everything, the smaller chunk wins
    pick = select_configuration([row(1024, 0.80, 0.5, 900), row(256, 0.80, 0.5, 900)])
    assert pick["chunk_target_tokens"] == 256


def test_retrieval_result_is_not_a_selection_until_the_run_completes(tmp_path, monkeypatch):
    """latest.json is the handoff to generation. An interrupted run must not leave
    one behind claiming a chunk size was chosen."""
    import json

    import retrieval_eval as rev

    monkeypatch.setattr(rev, "OUT_DIR", tmp_path)
    monkeypatch.setattr(rev, "LATEST", tmp_path / "latest.json")
    monkeypatch.setattr(rev, "LATEST_SMOKE", tmp_path / "latest-smoke.json")
    partial = {"complete": False, "configurations": [], "items": {}}
    path, latest = rev.write_result(partial, "stampA")
    assert latest is None and not (tmp_path / "latest.json").exists()
    assert json.loads(path.read_text())["complete"] is False

    good = {
        "timestamp": "t", "complete": True, "authoritative": True, "config": {},
        "config_digest": "cfg", "items": {},
        "configurations": [{"chunk_target_tokens": 512, "chunk_overlap_pct": 15,
                            "evidence_recall": 0.5, "evidence_hit_rate": 0.5,
                            "section_recall@3": 0.5, "mrr": 0.5, "ndcg@5": 0.5,
                            "context_precision": 0.5, "reranker_pair_truncated_pct": 0.3,
                            "embed_truncated_pct": None}],
        "selected": {"chunk_target_tokens": 512, "chunk_overlap_pct": 15},
        "selection_rule": rev.SELECTION_RULE, "canonical_case_ids": [], "golden_version": 2,
        "golden_digest": "d", "corpus_digest": "c", "prompt_digest": "p",
        "n_cases": 0, "excluded": [],
    }
    _, latest = rev.write_result(good, "stampB")
    assert latest == tmp_path / "latest.json"
    assert json.loads(latest.read_text())["complete"] is True
    assert not list(tmp_path.glob(".*.tmp")), "atomic write left its temp file behind"

    # a completed SMOKE run never replaces the authoritative selection
    smoke = json.loads(json.dumps(good)) | {"authoritative": False}
    smoke["configurations"][0]["chunk_target_tokens"] = 256
    smoke["selected"] = {"chunk_target_tokens": 256, "chunk_overlap_pct": 15}
    _, handoff = rev.write_result(smoke, "stampS")
    assert handoff == tmp_path / "latest-smoke.json"
    assert json.loads((tmp_path / "latest.json").read_text())["selected"]["chunk_target_tokens"] == 512

    # an out-of-range metric is refused rather than written
    bad = json.loads(json.dumps(good))
    bad["configurations"][0]["ndcg@5"] = 1.4
    assert any("outside [0, 1]" in e for e in rev.schema_errors(bad))
    with pytest.raises(SystemExit):
        rev.write_result(bad, "stampC")

    # a measured truncation share outside [0, 1] is refused as well
    bad = json.loads(json.dumps(good))
    bad["configurations"][0]["embed_truncated_pct"] = 3.0
    assert any("embed_truncated_pct" in e for e in rev.schema_errors(bad))

    # a selection naming a configuration that was never run is refused too
    bad = json.loads(json.dumps(good))
    bad["selected"] = {"chunk_target_tokens": 999, "chunk_overlap_pct": 15}
    assert any("not one of the configurations tested" in e for e in rev.schema_errors(bad))


def test_retrieval_evaluation_excludes_cases_with_no_gold_evidence():
    """Traps, declines and clarify cases retrieve nothing on purpose; averaging
    them in would report a retrieval failure that is actually correct behaviour."""
    from retrieval_eval import split_items

    rows = _golden_rows()
    scored, excluded = split_items(rows)
    assert len(scored) + len(excluded) == len(rows)
    assert all(r["answerable"] and r["context"] for r in scored)
    assert {e["id"] for e in excluded} >= {"amb-01", "amb-06", "trap-04", "unsup-02"}
    assert len(scored) == 43


# ==========================================================================
# Stage 2: generation evaluation (evals/generation_eval.py)
# ==========================================================================


def test_generation_refuses_a_stale_or_incomplete_retrieval_selection(tmp_path):
    """Answers generated at a chunk size nobody selected, or against gold labels
    that have since moved, are worse than no numbers at all."""
    import json

    import generation_eval as gev

    missing = tmp_path / "nope.json"
    with pytest.raises(SystemExit, match="no retrieval result"):
        gev.load_retrieval_selection(missing)

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"complete": False}), encoding="utf-8")
    with pytest.raises(SystemExit, match="did not finish"):
        gev.load_retrieval_selection(incomplete)

    current = {"complete": True, "authoritative": True,
               "golden_digest": gev.golden_digest(), "corpus_digest": gev.corpus_digest(),
               "prompt_digest": gev.prompt_digest(),
               "selected": {"chunk_target_tokens": 512, "chunk_overlap_pct": 15}}

    for key, match in (("golden_digest", "golden_digest"), ("corpus_digest", "corpus_digest"),
                       ("prompt_digest", "prompt_digest")):
        stale = tmp_path / f"stale-{key}.json"
        stale.write_text(json.dumps(current | {key: "0000000000000000"}), encoding="utf-8")
        with pytest.raises(SystemExit, match=match):
            gev.load_retrieval_selection(stale)

    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps(current | {"authoritative": False}), encoding="utf-8")
    with pytest.raises(SystemExit, match="smoke selection"):
        gev.load_retrieval_selection(smoke)
    assert gev.load_retrieval_selection(smoke, allow_smoke=True)["authoritative"] is False

    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(current), encoding="utf-8")
    assert gev.load_retrieval_selection(ok)["selected"]["chunk_target_tokens"] == 512


def test_result_paths_outside_the_repo_or_relative_do_not_crash(tmp_path, monkeypatch):
    """A relative --retrieval-result reached Path.relative_to(ROOT) unresolved and raised."""
    import retrieval_eval as rev

    assert rev.display_path(tmp_path / "x.json") == str((tmp_path / "x.json").resolve())
    monkeypatch.chdir(rev.ROOT)
    assert rev.display_path(Path("evals/golden_set.jsonl")) == str(Path("evals/golden_set.jsonl"))


def test_corpus_digest_sees_content_not_just_size(tmp_path, monkeypatch):
    """An equal-length edit to a source document must change the digest."""
    import index_corpus
    import retrieval_eval as rev
    doc = tmp_path / "doc.txt"
    doc.write_text("compensation is 600 EUR", encoding="utf-8")
    monkeypatch.setattr(index_corpus, "DATA", tmp_path)
    monkeypatch.setattr(index_corpus, "CORPUS", [{"doc_id": "d", "path": "doc.txt"}])
    before = rev.corpus_digest()
    doc.write_text("compensation is 400 EUR", encoding="utf-8")
    assert rev.corpus_digest() != before


def test_generation_freezes_retrieval_settings_from_the_selection():
    """Stage 2 must judge the retrieval that was selected, not whatever config.py says now."""
    from types import SimpleNamespace

    import generation_eval as gev

    base = Settings(_env_file=None, embedder="hash", reranker="none", embedding_dim=256)
    selection = {"selected": {"chunk_target_tokens": 512, "chunk_overlap_pct": 15},
                 "config": {"embedder": "hash", "reranker": "none", "embedding_dim": 256,
                            "final_k": 5, "candidates_k": 40}}
    args = SimpleNamespace(embedder=None, reranker=None)
    settings, frozen, overrides = gev.settings_for_selection(base, selection, args)
    assert (settings.chunk_target_tokens, settings.final_k, settings.candidates_k) == (512, 5, 40)
    assert frozen == {"final_k": (8, 5), "candidates_k": (20, 40)} and overrides == {}

    _, _, overrides = gev.settings_for_selection(
        base, selection, SimpleNamespace(embedder=None, reranker="ce"))
    assert overrides == {"reranker": "ce"}


def test_golden_check_detects_a_stale_saved_file(tmp_path):
    """--check used to validate a fresh build only, never the file both stages read."""
    import json

    import build_golden_set as bgs

    records = [{"id": "a", "question": "q1"}, {"id": "b", "question": "q2"}]
    saved = tmp_path / "golden.jsonl"
    saved.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    assert bgs.saved_file_drift(records, saved) == []

    changed = [records[0], {"id": "b", "question": "edited"}]
    assert bgs.saved_file_drift(changed, saved) == ["b: question"]
    assert "ids/order differ" in bgs.saved_file_drift(records[:1], saved)[0]


def _gen_row(rid, action, outcome, **kw):
    """A generation_eval.run_cases row with neutral defaults."""
    base = {"id": rid, "action": action, "answerable": action == "answer", "outcome": outcome,
            "generation_attempted": outcome in ("answered", "abstained", "validation_failed",
                                                "llm_error"),
            "citation_valid": True, "citation_coverage": None, "retry_count": 0,
            "clarified": outcome == "clarified", "asked_twice": False,
            "expects_unsupported": False, "flight_lookups": [], "supported_airline_sources": [],
            "unsupported_airline": None, "requires_flight": False, "got_flight_data": False,
            "response": "An answer [S1].", "model_answer": "An answer [S1].",
            "reference": "The reference.", "evidence": ["RETRIEVED SOURCE: x"],
            "user_input": "q", "gate": None, "prompt_tokens": None, "completion_tokens": None,
            "llm_calls": 0, "turn_s": 0.0, "context_tokens": 0}
    return {**base, **kw}


def test_ragas_adapter_maps_the_authorized_evidence_not_the_reference():
    """user_input / retrieved_contexts / response / reference, converted at runtime
    - there is no second golden CSV to drift from, and no gold passage in the evidence."""
    from generation_eval import to_ragas_dataset

    rows = [_gen_row("a", "answer", "answered", user_input="Was my flight from Paris delayed?",
                     evidence=["RETRIEVED SOURCE (sent to the model):\n[S1] EU261 Article 7"],
                     response="You may be owed compensation [S1].",
                     reference="EU261 applies on a departure from Paris.")]
    s = to_ragas_dataset(rows).samples[0]
    assert s.user_input == rows[0]["user_input"]
    assert s.retrieved_contexts == rows[0]["evidence"]
    assert s.response == rows[0]["response"] and s.reference == rows[0]["reference"]
    assert not s.reference_contexts


def test_generation_judges_only_faithfulness_and_factual_correctness():
    """Two LLM-judge metrics, and no others: no semantic similarity, no response
    relevancy, no LLM context precision/recall, no custom 1-5 rubric."""
    import asyncio

    import generation_eval as gev

    judge, name, _ = gev.build_offline_judge()
    text = "A passenger whose flight is cancelled is entitled to a refund."
    rows = [_gen_row("a", "answer", "answered", response=text, model_answer=text,
                     reference=text, evidence=[text])]
    asyncio.run(gev._score(rows, gev.judge_plan(rows), judge))
    scores = rows[0]["ragas"]
    assert {k for k in scores if not k.endswith("_basis")} == {
        "faithfulness", "factual_correctness", "errors"}
    assert scores["errors"] == []
    assert 0.0 <= scores["faithfulness"] <= 1.0 and 0.0 <= scores["factual_correctness"] <= 1.0
    assert name == "offline-lexical-stand-in"

    source = (ROOT / "evals" / "generation_eval.py").read_text(encoding="utf-8")
    for banned in ("SemanticSimilarity", "ResponseRelevancy", "AnswerRelevancy",
                   "LLMContextPrecision", "LLMContextRecall", "ContextRecall",
                   "ContextPrecision"):
        assert f"import {banned}" not in source and f"{banned}(" not in source


def test_api_judge_uses_an_async_client():
    """Regression: the Ragas collections metrics are async and call agenerate(),
    which an LLM built on a synchronous OpenAI client refuses outright - two
    errors per case and a run of nothing but n/a."""
    from generation_eval import JUDGE_REQUEST, build_api_judge

    llm, model, base_url, _ = build_api_judge("http://127.0.0.1:8000/v1", "local-judge",
                                              "not-needed", {"temperature": 0.0,
                                                             "max_completion_tokens": 8192})
    assert (model, base_url) == ("local-judge", "http://127.0.0.1:8000/v1")
    assert llm.is_async is True, "a sync client makes every agenerate() call fail"
    assert type(llm.client).__name__.startswith("Async")
    # a reasoning judge thinks before it emits the object; a truncated object is
    # an exception, not a low score, so the cap must be passed through
    assert llm.model_args["max_tokens"] == 8192
    assert llm.model_args["temperature"] == 0.0
    assert llm.client.max_retries >= 3        # a hosted judge's 429s are backed off, not errors
    # Session 19: the judge is Google AI Studio gemini-3.5-flash-lite. reasoning_effort
    # is gone on purpose - it is a Groq/OpenAI field and Gemini's OpenAI-compatibility
    # layer does not accept it.
    assert JUDGE_REQUEST == {"temperature": 0.0, "max_completion_tokens": 8192}
    assert "reasoning_effort" not in JUDGE_REQUEST


def test_judge_sends_exactly_the_frozen_request_and_meters_every_call():
    """On Groq the judge sends max_completion_tokens (not the deprecated max_tokens),
    the frozen temperature, JSON-object structured output and nothing of Ragas' own
    defaults; every response's usage is recorded."""
    import asyncio
    import json

    import httpx
    from generation_eval import JUDGE_REQUEST, build_api_judge, summarize_calls
    from pydantic import BaseModel

    class Verdict(BaseModel):
        verdict: int

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "openai/gpt-oss-120b",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"verdict": 1}'}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 300, "total_tokens": 1200,
                      "completion_tokens_details": {"reasoning_tokens": 250}}})

    llm, _, _, meter = build_api_judge("https://api.groq.com/openai/v1", "openai/gpt-oss-120b",
                                       "judge-key", JUDGE_REQUEST, transport=httpx.MockTransport(handler))
    assert asyncio.run(llm.agenerate("Return JSON with a verdict.", Verdict)).verdict == 1
    body = sent[0]
    assert body["max_completion_tokens"] == 8192 and "max_tokens" not in body and "top_p" not in body
    assert body["temperature"] == 0.0
    # nothing of the old Groq-only freeze leaks into the request any more
    assert "reasoning_effort" not in body
    assert body["response_format"] == {"type": "json_object"}
    assert meter.calls == [{"status": 200, "prompt_tokens": 900, "completion_tokens": 300,
                            "reasoning_tokens": 250, "finish_reason": "stop", "truncated": False}]
    s = summarize_calls(meter.calls, {"price_input_per_mtok": 0.15, "price_output_per_mtok": 0.60})
    assert s["cost_usd"] == round(900 * 0.15 / 1e6 + 300 * 0.60 / 1e6, 6) and s["calls"] == 1


def test_judge_on_a_non_groq_host_sends_the_cap_under_that_host_s_own_field():
    """Session 19: the judge is Google AI Studio through its OpenAI-compatibility
    endpoint. Groq deprecates max_tokens and wants max_completion_tokens; everyone
    else wants max_tokens, and Gemini rejects Groq's reasoning_effort outright."""
    import asyncio
    import json

    import httpx
    from generation_eval import JUDGE_EXPECTED_MODEL, JUDGE_REQUEST, build_api_judge
    from pydantic import BaseModel

    class Verdict(BaseModel):
        verdict: int

    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": JUDGE_EXPECTED_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"verdict": 1}'}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 300}})

    llm, _, _, _ = build_api_judge("https://generativelanguage.googleapis.com/v1beta/openai/",
                                   JUDGE_EXPECTED_MODEL, "judge-key", JUDGE_REQUEST,
                                   transport=httpx.MockTransport(handler))
    assert asyncio.run(llm.agenerate("Return JSON with a verdict.", Verdict)).verdict == 1
    body = sent[0]
    assert body["max_tokens"] == 8192 and "max_completion_tokens" not in body
    assert "reasoning_effort" not in body, "a Groq-only field would be rejected here"


def test_judge_client_paces_itself_under_a_requests_per_minute_cap():
    """AI Studio's free tier allows 15 requests/minute and one judged case makes
    several calls back to back, so the pacing must sit on the REQUEST, where
    instructor's re-asks and the SDK's own retries are counted too."""
    import asyncio

    from generation_eval import JUDGE_MIN_CALL_INTERVAL_S, CallPacer

    now = [0.0]
    slept = []

    async def sleep(s):
        slept.append(s)
        now[0] += s

    pacer = CallPacer(10.0, clock=lambda: now[0], sleep=sleep)

    async def three_calls_back_to_back():
        for _ in range(3):
            await pacer.on_request(object())

    asyncio.run(three_calls_back_to_back())
    assert slept == [10.0, 10.0], "every call after the first waits out the interval"
    # time already spent between calls counts against the interval, it is not added to it
    now[0] += 4.0
    asyncio.run(pacer.on_request(object()))
    assert slept[-1] == 6.0
    # and the shipped interval really does keep the run under 15 requests/minute
    assert 60 / JUDGE_MIN_CALL_INTERVAL_S <= 15


def test_judge_preflight_fails_before_any_candidate_token_is_spent():
    """Judging runs last, so an unreachable judge would otherwise be discovered only
    after a whole generation pass has been paid for - and on Groq's free tier that
    pass is most of a day's quota and cannot be replayed from disk."""
    import httpx
    import pytest
    from generation_eval import JUDGE_REQUEST, build_api_judge, preflight_judge

    def refuse(request):
        return httpx.Response(404, json={"error": {"message": "model not found", "code": 404}})

    llm, model, base_url, _ = build_api_judge("https://example.invalid/v1", "wrong-model-id",
                                              "judge-key", JUDGE_REQUEST,
                                              transport=httpx.MockTransport(refuse), max_retries=0)
    with pytest.raises(SystemExit) as raised:
        preflight_judge(llm, model, base_url)
    message = str(raised.value)
    assert "no tokens were spent on a candidate" in message
    assert "wrong-model-id" in message and "JUDGE_MODEL" in message


def test_stratified_sample_keeps_every_kind_of_case_and_repeats_on_its_seed():
    """A uniform draw of 10 from 43 answer / 4 clarify / 3 abstain cases leaves the
    clarify or abstain cases out about half the time - and those ARE the hard gates
    ('every preserved clarification and abstention edge case passes'). An empty gate
    passes vacuously, which reads as a success."""
    import collections
    import json

    from generation_eval import stratified_sample

    rows = [json.loads(line) for line in
            (ROOT / "evals" / "golden_set.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 50
    categories = {r["category"] for r in rows}
    for seed in range(10):
        drawn = stratified_sample(rows, 10, seed)
        assert len(drawn) == 10
        assert {r["category"] for r in drawn} == categories, "every category is represented"
        actions = collections.Counter(r["expected_behavior"]["action"] for r in drawn)
        assert actions["clarify"] >= 1 and actions["abstain"] >= 1, seed
        assert [r["id"] for r in drawn] == [r["id"] for r in rows if r["id"] in
                                            {d["id"] for d in drawn}], "golden order is kept"
    # the same seed is the same draw; different seeds generally are not
    assert ([r["id"] for r in stratified_sample(rows, 10, 7)]
            == [r["id"] for r in stratified_sample(rows, 10, 7)])
    assert ({r["id"] for r in stratified_sample(rows, 10, 1)}
            != {r["id"] for r in stratified_sample(rows, 10, 2)})


def test_deterministic_generation_metrics_come_from_outcomes_with_denominators():
    """A rate with no denominator is unreadable, and a decline is read from
    Answer.outcome - never from prose (a real answer can say 'does not cover')."""
    from generation_eval import deterministic_metrics

    rows = [
        _gen_row("a1", "answer", "answered", response="The sources do not cover X, but [S1]."),
        _gen_row("a2", "answer", "validation_failed", citation_valid=False),
        _gen_row("a3", "answer", "gated"),
        _gen_row("a4", "answer", "llm_error"),
        _gen_row("a5", "answer", "answered", retry_count=1),
        _gen_row("t1", "abstain", "gated"),
        _gen_row("t2", "abstain", "answered"),
        _gen_row("c1", "clarify", "clarified"),
        _gen_row("u1", "answer", "declined_unsupported", expects_unsupported=True,
                 unsupported_airline="BA"),
        _gen_row("u2", "answer", "answered", expects_unsupported=True, unsupported_airline="BA",
                 flight_lookups=["BA117"]),
    ]
    m = deterministic_metrics(rows)
    assert m["outcomes"]["answered"] == 4 and m["outcomes"]["gated"] == 2
    # citation validity and validation over turns where the model produced output
    # (a1, a2, a5, t2, u2 produced model output; a4 errored; the rest never generated)
    assert m["citation_validity_n"] == 5 and m["citation_validity_failed"] == ["a2"]
    assert m["validation_pass_rate"] == 0.8 and m["first_try_pass_rate"] == 0.6
    assert m["llm_error_rate"] == 0.1667 and m["llm_error_cases"] == ["a4"]
    assert m["abstention_accuracy"] == 0.5 and m["abstention_accuracy_failed"] == ["t2"]
    assert m["clarification_accuracy"] == 1.0 and m["clarification_accuracy_n"] == 1
    assert m["unsupported_airline_accuracy"] == 0.5
    assert m["unsupported_airline_accuracy_failed"] == ["u2"]
    # a1's prose declines part of the question but its outcome is "answered"
    assert m["over_abstention_cases"] == ["a3"] and m["over_abstention_rate_n"] == 7

    # no cases in a class -> None, never a flattering 1.0
    empty = deterministic_metrics([_gen_row("a1", "answer", "answered")])
    assert empty["abstention_accuracy"] is None and empty["abstention_accuracy_n"] == 0


def test_clarification_metric_requires_one_round_not_two():
    """Production may ask once. A case that is asked again after the reply has
    not clarified correctly, however good the eventual answer is."""
    from generation_eval import deterministic_metrics

    m = deterministic_metrics([_gen_row("c1", "clarify", "clarified"),
                               _gen_row("c2", "clarify", "answered"),
                               _gen_row("c3", "clarify", "clarified", asked_twice=True)])
    assert m["clarification_accuracy_n"] == 3
    assert m["clarification_accuracy_failed"] == ["c2", "c3"]


def test_generation_eval_surfaces_the_documented_over_clarification_gap():
    """CLAUDE.md records 6 cases the golden set says are answerable as asked but
    production still asks about. The final answer hides that (the scripted reply
    resolves it), so the harness replays the first turn - otherwise a known
    routing gap would score as a pass."""
    from generation_eval import deterministic_metrics
    from golden.replay import replay

    asked = [r["id"] for r in _golden_rows()
             if (r["expected_behavior"]["action"] == "answer"
                 and replay(r["question"], r.get("reply"), r.get("flight_fixture"))[0].clarification)]
    assert asked == ["dom-24", "dom-31", "dom-37", "dom-44", "us-eur-09", "law-02"]

    rows = [_gen_row(i, "answer", "answered", clarified=i in asked) for i in [*asked, "dom-01"]]
    assert deterministic_metrics(rows)["answer_cases_production_asked"] == asked


def test_no_answer_delivered_scores_zero_and_is_never_dropped():
    """An answer-required case that was gated, failed validation or errored used to
    vanish from the Ragas mean, so a failure improved the average."""
    import asyncio

    import generation_eval as gev

    rows = [_gen_row("ok", "answer", "answered"),
            _gen_row("gated", "answer", "gated", evidence=[]),
            _gen_row("invalid", "answer", "validation_failed"),
            _gen_row("error", "answer", "llm_error", model_answer=""),
            _gen_row("ask", "clarify", "clarified", evidence=[]),
            _gen_row("trap", "abstain", "answered")]
    plan = {p["id"]: p for p in gev.judge_plan(rows)}
    assert plan["ok"]["judge_faithfulness"] and plan["ok"]["judge_factual"]
    for rid in ("gated", "invalid", "error"):
        assert not plan[rid]["judge_faithfulness"] and not plan[rid]["judge_factual"]
        assert plan[rid]["factual_basis"].startswith("no answer delivered")
    assert plan["ask"]["factual_basis"].startswith("not applicable")
    # an answered trap is a shown answer: its faithfulness is judged too
    assert plan["trap"]["judge_faithfulness"] and not plan["trap"]["judge_factual"]

    judge, _, _ = gev.build_offline_judge()
    asyncio.run(gev._score(rows, gev.judge_plan(rows), judge))
    s = gev.ragas_summary(rows)
    assert s["factual_correctness_required_n"] == 4 and s["factual_correctness_n"] == 4
    assert sorted(s["factual_correctness_no_answer_cases"]) == ["error", "gated", "invalid"]
    assert s["factual_correctness_judged_only_n"] == 1
    assert s["faithfulness_applicable_n"] == 2          # ok + trap: the shown generated answers


def test_faithfulness_evidence_is_exactly_what_generation_was_authorized_to_use():
    """Source blocks with their headers exactly as sent, the notes and flight block
    sent with them, and the substantive prompt rules - never the reference, and
    no rebuilt or re-ordered context."""
    from types import SimpleNamespace

    import generation_eval as gev

    from flight_delay.generation import SYSTEM_PROMPT, flight_block

    live = _flight("CDG", "JFK")
    chunks = [_chunk("c1", "A refund is owed when the carrier cancels. --- not a separator"),
              _chunk("c2", "Compensation of EUR 600 applies to flights over 3500 km.")]
    built = build_context(chunks, live, token_budget=5000, reserve_for_answer=0)
    note = "ROUTE ASSUMPTION: Assuming you're flying from Paris. Answer on this assumption."
    answer = SimpleNamespace(built_context=built, diagnostics={"prompt_notes": [note]})
    guidance = gev.prompt_guidance()
    evidence = gev.authorized_evidence(answer, guidance)

    sources = [e for e in evidence if e.startswith("RETRIEVED SOURCE")]
    assert len(sources) == len(built.source_map) == 2
    for marker, block in zip(built.source_map, gev.source_blocks(built), strict=True):
        assert block.startswith(f"[{marker}] (LAW")            # header kept, order as sent
        assert block in built.text
    supplied = next(e for e in evidence if e.startswith("SUPPLIED WITH THE QUESTION"))
    assert note in supplied and flight_block(live) in supplied
    assert evidence[-1].startswith("SYSTEM PROMPT GUIDANCE")
    for rule in ("RULE 1:", "RULE 2:", "RULE 4:", "RULE 5:"):
        assert rule in guidance
    for rule in ("RULE 3:", "RULE 6:", "RULE 7:"):
        assert rule not in guidance
    # read from the prompt file, whole rules, not paraphrased
    assert all(part.strip() in SYSTEM_PROMPT for part in guidance.split("\n\nRULE "))

    # nothing generated -> nothing authorized
    assert gev.authorized_evidence(SimpleNamespace(built_context=None, diagnostics={}), guidance) == []
    with pytest.raises(SystemExit, match="not found"):
        gev.prompt_guidance("no rules here")


def test_pipeline_records_what_generation_was_given_on_every_outcome():
    """Stage 2 reads the gate decision, the exact prompt notes and token usage from
    the Answer; the notes recorded are the ones the model received."""
    from flight_delay.generation import Completion

    class UsageLLM(_ScriptedLLM):
        def complete(self, system, user):
            text = super().complete(system, user)
            self.last = Completion(text, "stop", 1200, 80, 1)
            return text

    llm = UsageLLM("A refund is owed when the airline cancels [S1].")
    llm.last = Completion("stale", "stop", 999_999, 999_999, 1)   # must not be reported
    pipe = _run_pipeline(llm, confidence_gate_enabled=True, min_rerank_score=-100.0,
                         min_score_margin=0.0, min_grounding_ratio=0.0)
    # ask once, get no details, answer on a stated assumption: a ROUTE ASSUMPTION note
    assert pipe.run("My American flight was cancelled, can I get a refund?",
                    conversation_id="c1").outcome == "clarified"
    assert llm.last.prompt_tokens == 999_999                      # untouched: no model call yet
    a = pipe.run("Sorry, I don't have my flight number or the airport details.",
                 conversation_id="c1")
    assert a.outcome == "answered" and a.assumption
    assert a.diagnostics["prompt_notes"][0].startswith(f"ROUTE ASSUMPTION: {a.assumption}")
    assert a.diagnostics["gate"]["should_answer"] is True and "signals" in a.diagnostics["gate"]
    assert a.diagnostics["usage"] == [{"prompt_tokens": 1200, "completion_tokens": 80,
                                       "reasoning_tokens": None, "reasoning_chars": 0,
                                       "finish_reason": "stop", "truncated": False,
                                       "max_completion_tokens": None, "attempts": 1, "error": None}]
    assert a.diagnostics["model_answer"] == "A refund is owed when the airline cancels [S1]."
    for note in a.diagnostics["prompt_notes"]:
        assert f"\n\n{note}" in llm.prompts[0]


def test_gate_calibration_counts_false_and_missed_abstentions_per_threshold():
    """Calibration reads the recorded signals; clarify cases count as neither."""
    import generation_eval as gev

    def gated(rid, action, top, ratio, answers):
        return _gen_row(rid, action, "answered" if answers else "gated",
                        gate={"should_answer": answers, "reasons": [] if answers else ["x"],
                              "signals": {"n_chunks": 6.0, "rerank_top": top,
                                          "score_margin": 3.0, "grounding_ratio": ratio}})

    rows = [gated("a1", "answer", 1.0, 0.8, True), gated("a2", "answer", -3.0, 0.5, False),
            gated("a3", "answer", 0.5, 0.2, False), gated("t1", "abstain", -1.0, 0.1, True),
            _gen_row("t2", "abstain", "declined_unsupported"), _gen_row("c1", "clarify", "clarified")]
    settings = Settings(_env_file=None, embedder="hash", reranker="none")
    cal = gev.gate_calibration(rows, settings)
    assert cal["false_abstentions"] == ["a2", "a3"] and cal["missed_abstentions"] == ["t1"]
    assert cal["abstain_cases_not_reaching_gate"] == ["t2"]
    grid = {(g["min_rerank_score"], g["min_score_margin"], g["min_grounding_ratio"]): g
            for g in cal["pareto_front"]}
    assert all(g["false_abstentions"] >= 0 for g in grid.values())
    # permissive: nothing refused, the trap gets through; grounding 0.2 catches only the trap
    assert gev.gate_would_answer(rows[3]["gate"]["signals"], None, 0.0, 0.0)
    assert not gev.gate_would_answer(rows[3]["gate"]["signals"], None, 0.0, 0.2)
    assert gev.gate_would_answer(rows[2]["gate"]["signals"], None, 0.0, 0.2)
    assert not gev.gate_would_answer({"n_chunks": 0.0}, None, 0.0, 0.0)
    assert any(g["false_abstentions"] == 0 and g["missed_abstentions"] == 0
               for g in cal["pareto_front"])


def _summary(faith=0.9, fc=0.5, err=0.0, tokens=1000, secs=1.0, cost=0.001, judge_errors=0, **kw):
    s = {"faithfulness": faith, "factual_correctness": fc, "llm_error_rate": err,
         "generation_cases": 40, "tokens_per_case": tokens, "generation_s_mean": secs,
         "cost_per_case_usd": cost, "judged_calls_planned": 40, "judge_errors": judge_errors,
         "citation_validity_failed": [], "scope_safety_failed": [],
         "clarification_accuracy_n": 4, "clarification_accuracy_failed": [],
         "abstention_accuracy_n": 3, "abstention_accuracy_failed": []}
    return s | kw


def test_bake_off_recommendation_is_the_documented_rule():
    import generation_eval as gev

    def cand(name, **kw):
        return {"name": name, "summary": _summary(**kw)}

    # faithfulness is a floor (within 0.05 of best), then factual correctness decides
    rec = gev.recommend_generator([cand("a", faith=0.90, fc=0.40), cand("b", faith=0.86, fc=0.55),
                                   cand("c", faith=0.70, fc=0.90)])
    assert rec["recommended"] == "b" and rec["within_faithfulness_tolerance"] == ["a", "b"]
    # every hard gate excludes a candidate however well it scores
    for failing in ({"err": 0.06}, {"judge_errors": 5}, {"citation_validity_failed": ["dom-01"]},
                    {"scope_safety_failed": ["trap-08"]}, {"clarification_accuracy_failed": ["amb-01"]},
                    {"abstention_accuracy_failed": ["trap-04"]}, {"abstention_accuracy_n": 0},
                    {"faithfulness": None}):
        rec = gev.recommend_generator([cand("a", faith=0.80, fc=0.40),
                                       cand("b", **({"faith": 0.95, "fc": 0.90} | failing))])
        assert rec["recommended"] == "a" and rec["ineligible"]["b"], failing
    # generation errors at exactly 5% and judge errors at exactly 10% still pass
    assert not gev.hard_gate_failures(_summary(err=0.05, judge_errors=4))
    # nobody passes the hard gates -> no winner
    rec = gev.recommend_generator([cand("a", err=0.5), cand("b", judge_errors=40)])
    assert rec["recommended"] is None and "no winner" in rec["reason"]
    # ties on FactualCorrectness -> measured cost/case, then tokens/case, then latency
    rec = gev.recommend_generator([cand("a", cost=0.002, tokens=100), cand("b", cost=0.001, tokens=9000)])
    assert rec["recommended"] == "b"
    rec = gev.recommend_generator([cand("a", tokens=9000), cand("b", tokens=4000)])
    assert rec["recommended"] == "b"
    rec = gev.recommend_generator([cand("a", secs=9.0), cand("b", secs=2.0)])
    assert rec["recommended"] == "b"
    # --pilot never selects: two fixed cases are for checking wiring (user, Session 14)
    rec = gev.recommend_generator([cand("a")], full_run=False, wiring_only=True,
                                  basis="a SUBSET of 2 of 50")
    assert rec["recommended"] is None and "wiring only" in rec["reason"]
    # Session 19: any OTHER subset may now select, so a generator can be adopted
    # without the full bake-off Groq's free tier cannot finish - but it must say what
    # the choice rests on, and must not claim to be a full run.
    rec = gev.recommend_generator([cand("a", faith=0.90, fc=0.40), cand("b", faith=0.86, fc=0.55)],
                                  full_run=False, basis="a SUBSET of 10 of 50 golden cases")
    assert rec["recommended"] == "b"
    assert rec["full_run"] is False and "SUBSET of 10 of 50" in rec["basis"]
    # the hard gates still apply on a subset: a subset cannot promote a failing candidate
    rec = gev.recommend_generator([cand("a", err=0.5)], full_run=False, basis="subset")
    assert rec["recommended"] is None and "no winner" in rec["reason"]
    # a full run says so, and carries the basis too
    rec = gev.recommend_generator([cand("a")], basis="the full canonical golden set (50 cases)")
    assert rec["full_run"] is True and "50 cases" in rec["basis"]


def test_winner_production_settings_state_the_tested_completion_cap(monkeypatch):
    import generation_eval as gev

    monkeypatch.setattr(gev, "_dotenv", lambda name: {"GROQ_API_KEY": "k"}.get(name))
    candidates = gev.load_candidates()
    settings = Settings(_env_file=None, embedder="hash", reranker="none")
    rec = gev.recommend_generator(
        [{"name": "groq-gpt-oss-20b", "summary": _summary(), "generator": candidates["groq-gpt-oss-20b"]}],
        settings)
    env = rec["production_settings"]
    assert env["LLM_MAX_COMPLETION_TOKENS"] == 900 and env["CONTEXT_ANSWER_RESERVE_TOKENS"] == 700
    assert env["LLM_CONTEXT_WINDOW"] == 8192 and env["LLM_EXTRA_BODY"] == '{"reasoning_effort": "low"}'
    assert env["LLM_RETRY_MAX_WAIT_S"] == 120
    # Session 19: request pacing is an evaluation-harness control and never a
    # production setting, even for a candidate that needed it to be measured at all
    # (gpt-oss-20b now paces at 45s, qwen at 60s, on Groq's free tier). A served app
    # answers one question at a time; a burst is handled by LLM_MAX_RETRIES.
    assert candidates["groq-gpt-oss-20b"]["min_call_interval_s"] == 45
    assert "LLM_MIN_CALL_INTERVAL_S" not in env
    qwen = gev.production_settings(candidates["groq-qwen3.8-27b"], settings)
    assert "LLM_MIN_CALL_INTERVAL_S" not in qwen and qwen["LLM_MAX_COMPLETION_TOKENS"] == 900
    # and production refuses a reasoning configuration that would fall back to the reserve
    with pytest.raises(ValueError, match="LLM_MAX_COMPLETION_TOKENS is unset"):
        Settings(_env_file=None, llm_extra_body={"reasoning_effort": "low"})
    ok = Settings(_env_file=None, llm_extra_body={"reasoning_effort": "low"}, llm_max_completion_tokens=4096)
    assert ok.completion_cap_tokens == 4096


def test_answer_reserve_is_renamed_and_the_old_name_still_works(monkeypatch):
    import argparse

    import generation_eval as gev

    from flight_delay.config import revalidate

    monkeypatch.setenv("LLM_MAX_TOKENS", "650")
    s = Settings(_env_file=None)
    assert s.context_answer_reserve_tokens == 650 and s.llm_max_tokens == 650
    assert s.completion_cap_tokens == 650          # no cap set and no reasoning configured
    monkeypatch.delenv("LLM_MAX_TOKENS")
    monkeypatch.setenv("CONTEXT_ANSWER_RESERVE_TOKENS", "600")
    assert Settings(_env_file=None).context_answer_reserve_tokens == 600
    assert revalidate(Settings(_env_file=None)).context_answer_reserve_tokens == 600
    monkeypatch.delenv("CONTEXT_ANSWER_RESERVE_TOKENS")
    assert Settings(_env_file=None, llm_max_tokens=500).context_answer_reserve_tokens == 500

    # a Stage 1 selection written before the rename still freezes the reserve
    base = Settings(_env_file=None, embedder="hash", reranker="none", llm_max_tokens=900)
    selection = {"selected": {"chunk_target_tokens": 512, "chunk_overlap_pct": 15},
                 "config": {"llm_max_tokens": 700, "llm_context_window": 8192}}
    frozen, diffs, _ = gev.settings_for_selection(base, selection,
                                                  argparse.Namespace(embedder=None, reranker=None))
    assert frozen.context_answer_reserve_tokens == 700 and diffs["context_answer_reserve_tokens"] == (900, 700)


def test_candidates_file_holds_no_keys_and_resolves_keys_by_name(monkeypatch):
    import generation_eval as gev

    raw = gev.CANDIDATES_FILE.read_text(encoding="utf-8")
    assert "gsk_" not in raw and '"api_key"' not in raw
    candidates = gev.load_candidates()
    # the local Ollama candidate was retired after the first pilot (Session 15)
    assert not any("ollama" in name or c["provider"] == "ollama" for name, c in candidates.items())
    assert "ollama" not in raw.lower() and "11434" not in raw
    groq = candidates["groq-gpt-oss-20b"]

    # generator calls use GROQ_API_KEY only: never the judge's key, even on the judge's host
    monkeypatch.setattr(gev, "_dotenv", lambda name: {
        "GROQ_API_KEY": None, "JUDGE_BASE_URL": "https://api.groq.com/openai/v1",
        "JUDGE_API_KEY": "judge-secret"}.get(name))
    with pytest.raises(SystemExit, match="never use JUDGE_API_KEY"):
        gev.candidate_api_key(groq)
    assert gev.judge_api_key("https://api.groq.com/openai/v1") == "judge-secret"
    monkeypatch.setattr(gev, "_dotenv", lambda name: {"GROQ_API_KEY": "gen-secret"}.get(name))
    assert gev.candidate_api_key(groq) == ("gen-secret", "GROQ_API_KEY")
    with pytest.raises(SystemExit, match="needs JUDGE_API_KEY"):
        gev.judge_api_key("https://api.groq.com/openai/v1")
    local = {"name": "local-vllm", "provider": "vllm", "base_url": "http://127.0.0.1:8000/v1", "model": "m"}
    assert gev.candidate_api_key(local)[0] == "not-needed-for-local"

    # every candidate must accept the same frozen input assembly plus its own cap
    settings = Settings(_env_file=None, embedder="hash", reranker="none", llm_context_window=16384)
    with pytest.raises(SystemExit, match="below the largest assembled prompt"):
        gev.generator_settings(settings, dict(groq, context_window=12000))


def test_candidate_completion_cap_is_sent_but_evidence_stays_sized_by_the_frozen_reserve(monkeypatch):
    """Each Groq candidate gets the 900-token completion cap (sent as
    max_completion_tokens) and its reasoning_effort, while the frozen input-assembly
    budget and answer reserve size the evidence identically; the served window must
    hold the largest assembled prompt plus the cap. Qwen 3.8 is paced 60 s apart."""
    import json

    import generation_eval as gev
    import httpx

    from flight_delay.generation import build_llm, evidence_token_budget

    monkeypatch.setattr(gev, "_dotenv", lambda name: {"GROQ_API_KEY": "k"}.get(name))
    candidates = gev.load_candidates()
    settings = Settings(_env_file=None, embedder="hash", reranker="none", llm_context_window=8192)
    budgets = set()
    # Session 19: gpt-oss-20b paces at 45 s too. Groq's free tier refills TPM 8000 at
    # ~133 tokens/s and one call costs ~5,700, so back-to-back calls 429 every time.
    for name, effort, interval in (("groq-gpt-oss-20b", "low", 45.0), ("groq-qwen3.8-27b", "none", 60.0)):
        gs = gev.generator_settings(settings, candidates[name])
        assert gs.context_answer_reserve_tokens == 700 and gs.llm_max_completion_tokens == 900
        assert gs.llm_min_call_interval_s == interval and gs.llm_retry_max_wait_s == 120
        budgets.add(evidence_token_budget(gs, "my flight was cancelled", None))
        llm = build_llm(gs)
        assert llm.min_call_interval_s == interval
        seen = {}

        def handler(request, seen=seen):
            seen.update(json.loads(request.content))
            return _ok(usage={"prompt_tokens": 5000, "completion_tokens": 300})

        llm._transport = httpx.MockTransport(handler)
        completion = llm.generate("sys", "user")
        assert seen["max_completion_tokens"] == 900 and "max_tokens" not in seen, name
        assert seen["reasoning_effort"] == effort and seen["temperature"] == 0.0
        assert completion.max_completion_tokens == 900 and not completion.truncated
    assert budgets == {evidence_token_budget(settings, "my flight was cancelled", None)}

    # a served window must hold the largest prompt (8192 - 700) plus the 900 cap
    small = dict(candidates["groq-gpt-oss-20b"], context_window=8391)
    with pytest.raises(SystemExit, match="below the largest assembled prompt"):
        gev.generator_settings(settings, small)
    gev.generator_settings(settings, dict(small, context_window=8392))
    too_big = dict(candidates["groq-qwen3.8-27b"], max_completion_tokens=20000)
    with pytest.raises(SystemExit, match="exceeds the provider's limit"):
        gev.generator_settings(settings, too_big)
    with pytest.raises(ValueError, match="llm_max_completion_tokens"):
        Settings(_env_file=None, llm_max_completion_tokens=500)

    # measured per call: prompt_tokens + cap against the served window
    calls = gev.check_calls([{"prompt_tokens": 8000, "max_completion_tokens": 4096},
                             {"prompt_tokens": 9000, "max_completion_tokens": 4096},
                             {"prompt_tokens": None, "max_completion_tokens": 4096}], 12288)
    assert [c["fits_served_window"] for c in calls[:2]] == [True, False]
    assert calls[2]["fits_served_window"].startswith("unchecked")


def test_pilot_uses_fixed_case_ids_and_extrapolates_measured_cost():
    import argparse

    import generation_eval as gev

    rows = [{"id": i} for i in ("dom-01", "dom-04", "uk-us-01", "trap-04", "amb-01")]

    def args(**kw):
        return argparse.Namespace(**({"pilot": False, "cases": None, "limit": None} | kw))

    assert [r["id"] for r in gev.select_rows(rows, args(pilot=True))] == ["uk-us-01", "trap-04"]
    assert gev.PILOT_CASE_IDS == ("uk-us-01", "trap-04")
    assert [r["id"] for r in gev.select_rows(rows, args(cases="amb-01,dom-04"))] == ["dom-04", "amb-01"]
    with pytest.raises(SystemExit, match="unknown golden case ids"):
        gev.select_rows(rows, args(cases="nope"))
    with pytest.raises(SystemExit, match="only one"):
        gev.select_rows(rows, args(pilot=True, limit=2))
    # the pilot ids exist in the real golden set: one answer case and one abstain edge case
    import retrieval_eval

    golden = {r["id"]: r for r in retrieval_eval.load_golden_rows()}
    assert golden["uk-us-01"]["answerable"] and "cross_jurisdiction" in golden["uk-us-01"]["tags"]
    assert not golden["trap-04"]["answerable"] and "out_of_scope" in golden["trap-04"]["tags"]

    routing = ([{"generation_attempted": True, "outcome": "llm_error"}] * 30
               + [{"generation_attempted": False, "outcome": "gated"}] * 10
               + [{"generation_attempted": False, "outcome": "clarified"}] * 4)
    cands = [{"name": "g", "summary": {"cost_per_case_usd": 0.01, "judge_cost_per_judged_case_usd": 0.02}},
             {"name": "local", "summary": {"cost_per_case_usd": 0.0, "judge_cost_per_judged_case_usd": 0.03}}]
    est = gev.estimate_full_run(routing, cands, "off", 2)
    assert est["full_run_generation_cases"] == 40
    assert est["per_candidate"]["g"]["total_usd"] == 1.2 and est["total_usd"] == 2.4
    assert gev.estimate_full_run(routing, cands, "configured", 2)["full_run_generation_cases"] == 30


def test_pipeline_records_a_failed_model_call():
    from flight_delay.generation import LLMTransientError

    class Failing:
        last = None
        max_tokens = 4096

        def complete(self, system, user):  # noqa: ARG002
            raise LLMTransientError("groq still failing after 4 attempts: HTTP 429")

    pipe = _run_pipeline(Failing())
    a = pipe.run("Can I get a refund if American cancels my domestic flight?", conversation_id="c1")
    assert a.outcome == "llm_error"
    [call] = a.diagnostics["usage"]
    assert call["error"].startswith("LLMTransientError") and call["prompt_tokens"] is None
    assert call["max_completion_tokens"] == 4096


def test_pipeline_keeps_the_providers_complete_error():
    from flight_delay.generation import LLMRequestError

    long_message = "Request too large for model `m` in organization `org_SECRET1` " + "y" * 900

    class Refused:
        last = None
        max_tokens = 900

        def complete(self, system, user):  # noqa: ARG002
            raise LLMRequestError(f"groq rejected the request: HTTP 429 [rate_limit_exceeded]: {long_message}",
                                  status=429, provider_code="rate_limit_exceeded", provider_type="tokens",
                                  provider_message=long_message, retry_after_s=12.0)

    a = _run_pipeline(Refused()).run("Can I get a refund if American cancels my domestic flight?",
                                     conversation_id="c1")
    [call] = a.diagnostics["usage"]
    assert (call["error_status"], call["error_code"], call["retry_after_s"]) == (429, "rate_limit_exceeded", 12.0)
    assert call["error"].endswith("y" * 900) and "org_SECRET1" not in call["error"]
    d = a.diagnostics
    assert d["llm_error_detail"].endswith("y" * 900) and d["llm_error_status"] == 429
    assert d["llm_error_code"] == "rate_limit_exceeded" and d["llm_error_retry_after_s"] == 12.0


def test_llm_records_reasoning_tokens_when_reported():
    usage = {"prompt_tokens": 10, "completion_tokens": 300,
             "completion_tokens_details": {"reasoning_tokens": 250}}
    client, _ = _llm(lambda request: _ok(usage=usage))
    completion = client.generate("sys", "user")
    assert completion.reasoning_tokens == 250 and completion.completion_tokens == 300
    client, _ = _llm(lambda request: _ok(usage={"prompt_tokens": 1, "completion_tokens": 2}))
    assert client.generate("sys", "user").reasoning_tokens is None


def test_remote_calls_are_refused_without_explicit_confirmation(tmp_path, monkeypatch):
    """The run prints its plan and stops before building a judge, loading a model or
    opening a connection when any endpoint is a remote API."""
    import json

    import generation_eval as gev
    import httpx

    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "complete": True, "authoritative": True, "golden_digest": gev.golden_digest(),
        "corpus_digest": gev.corpus_digest(), "prompt_digest": gev.prompt_digest(),
        "selected": {"chunk_target_tokens": 512, "chunk_overlap_pct": 15},
        "config": {"embedder": "hash", "reranker": "none", "embedding_dim": 256}}), encoding="utf-8")

    def explode(*a, **kw):
        raise AssertionError("a network call was attempted")

    monkeypatch.setattr(httpx.Client, "request", explode)
    monkeypatch.setattr(gev, "build_judge", explode)
    monkeypatch.setattr(gev, "_dotenv", lambda name: {
        "JUDGE_BASE_URL": "https://api.groq.com/openai/v1", "JUDGE_MODEL": "judge-model"}.get(name))
    with pytest.raises(SystemExit, match="Nothing was sent"):
        gev.main(["--retrieval-result", str(selection), "--backend", "memory",
                  "--generator", "groq-llama-3.1-8b-instant"])
    # a local generator with a remote judge is still a paid run
    local = tmp_path / "candidates.json"
    local.write_text(json.dumps({"candidates": [{
        "name": "local-vllm", "provider": "vllm", "base_url": "http://127.0.0.1:8000/v1",
        "model": "m"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="judge judge-model"):
        gev.main(["--retrieval-result", str(selection), "--backend", "memory",
                  "--candidates", str(local), "--generator", "local-vllm"])


def test_only_a_production_equivalent_bake_off_is_authoritative():
    import generation_eval as gev

    groq = {"name": "g", "provider": "groq"}
    ok = {"cli_overrides": {}, "backend": "postgres", "limit": None, "gate": "configured",
          "judge": "api", "self_judged": [], "generators": [groq]}
    selection = {"authoritative": True}
    assert gev.run_is_authoritative(selection, **ok)
    for change in ({"gate": "off"}, {"limit": 2}, {"backend": "memory"}, {"judge": "fake"},
                   {"self_judged": ["g"]}, {"cli_overrides": {"reranker": "none"}},
                   {"generators": []}, {"generators": [{"name": "echo", "provider": "echo"}]}):
        assert not gev.run_is_authoritative(selection, **(ok | change)), change
    assert not gev.run_is_authoritative({"authoritative": False}, **ok)


def test_retrieval_is_shared_by_every_candidate():
    import generation_eval as gev

    class Inner:
        calls = 0

        def retrieve(self, query, **kw):
            Inner.calls += 1
            return object()

    r = gev.CachingRetriever(Inner())
    a = r.retrieve("q", flt={"airline_scope": "UA"})
    assert r.retrieve("q", flt={"airline_scope": "UA"}) is a
    assert r.retrieve("q", top_k=3, flt={"airline_scope": "UA"}) is not a
    assert Inner.calls == 2 == r.misses


def test_evaluation_never_reaches_the_real_airlabs_tool():
    """Both stages must be deterministic and free. A live lookup on an invented
    99xx flight number would also spend the monthly quota on nothing."""
    import httpx
    from golden.replay import FixtureTool, run_conversation

    from flight_delay.tools import AirLabsTool

    def explode(*a, **kw):
        raise AssertionError("evaluation opened an HTTP connection")

    pipe, _ = _pipeline()
    rows = _golden_rows()
    saved = httpx.Client.request
    httpx.Client.request = explode
    try:
        for item in rows[:12]:
            ans = run_conversation(pipe, item)
            assert ans is not None
    finally:
        httpx.Client.request = saved

    # FixtureTool is not AirLabsTool, and has no network method at all
    tool = FixtureTool(None)
    assert not isinstance(tool, AirLabsTool)
    assert not hasattr(tool, "_fetch_live")
    for name in ("retrieval_eval.py", "generation_eval.py"):
        source = (ROOT / "evals" / name).read_text(encoding="utf-8")
        assert "build_tool" not in source and "AirLabsTool" not in source


def test_unsupported_airlines_are_never_looked_up_or_given_another_carriers_policy():
    """BA117 gets no AirLabs call and no American/Delta/United/Southwest document:
    scoping it to its own code means regulations only, since the corpus has none."""
    from golden.replay import FixtureTool, run_conversation

    from flight_delay.tools import SUPPORTED_AIRLINES

    rows = [r for r in _golden_rows() if r.get("unsupported_airline")]
    assert {r["id"] for r in rows} == {"trap-08", "unsup-01", "unsup-02"}
    pipe, retriever = _pipeline()
    for item in rows:
        tool = FixtureTool(item.get("flight_fixture"))
        ans = run_conversation(pipe, item, tool=tool)
        assert tool.calls == [], f"{item['id']} looked up {tool.calls}"
        assert ans.live_data is None, item["id"]
        assert ans.unsupported_airline == item["unsupported_airline"], item["id"]
        used = {c.airline_iata for c in ans.context_chunks} & SUPPORTED_AIRLINES
        assert not used, f"{item['id']} answered from {used}"


def test_hybrid_cases_receive_their_synthetic_flight_record():
    """The six live-NN cases exist to test the flight-data path; without their
    fixture they silently become ordinary policy questions."""
    from golden.replay import FixtureTool, run_conversation

    hybrids = [r for r in _golden_rows() if r.get("flight_fixture")]
    assert len(hybrids) == 6
    pipe, _ = _pipeline()
    for item in hybrids:
        tool = FixtureTool(item["flight_fixture"])
        ans = run_conversation(pipe, item, tool=tool)
        assert tool.calls == [item["flight_fixture"]["flight_iata"]], item["id"]
        assert ans.live_data is not None and ans.live_data.dep_iata == item["flight_fixture"]["dep_iata"]


def test_removed_eval_entry_points_are_gone_and_unreferenced():
    """Two stages, no parallel old path: a second harness is how two sets of
    numbers start disagreeing about the same system."""
    evals = ROOT / "evals"
    for gone in ("chunk_sweep.py", "run.py", "judge.py", "compare.py", "ablate.py",
                 "metrics.py", "ablation_raw.json"):
        assert not (evals / gone).exists(), gone
    assert not (evals / "golden" / "retired_cases.py").exists()
    assert sorted(p.name for p in evals.glob("*.py")) == [
        "build_golden_set.py", "generation_eval.py", "retrieval_eval.py"]

    # This test file is excluded on purpose: it has to name the removed files in
    # order to assert they are gone. PROGRESS.md is excluded too - it is a
    # historical log, and rewriting history to match today's layout is a lie.
    # README.md is not an input here: it is optional overview text (user requirement).
    live = [ROOT / "Makefile", ROOT / "RUNBOOK.md", ROOT / "RESULTS.md",
            ROOT / "CLAUDE.md", ROOT / "evals" / "retrieval_eval.py",
            ROOT / "evals" / "generation_eval.py", ROOT / "evals" / "build_golden_set.py",
            ROOT / "evals" / "golden" / "replay.py", ROOT / "src" / "flight_delay" / "config.py",
            ROOT / "deploy" / "prometheus" / "rules" / "rag.yml"]
    for f in live:
        if f in LOCAL_ONLY_DOCS and not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        for stale in ("chunk_sweep", "evals/run.py", "evals/judge.py", "evals/compare.py",
                      "make ablate", "make compare"):
            assert stale not in text, f"{f.name} still refers to {stale}"


def test_tokenizer_preserves_legal_tokens():
    """'$1,075' and '250.5' must survive tokenisation or sparse retrieval loses
    exactly the signal it exists to provide."""
    toks = tokenize("Compensation of $1,075 under § 250.5")
    assert any("250.5" in t for t in toks)
    assert any("$1" in t for t in toks)


def test_settings_are_env_driven():
    s = Settings(embedder="hash", final_k=4)
    assert s.embedder == "hash" and s.final_k == 4


@pytest.mark.parametrize("bad", [
    {"final_k": 60, "candidates_k": 50},                        # final pool larger than candidates
    {"final_k": 3},                                             # 2 + 2 source quotas cannot fit
    {"chunk_overlap_pct": 80},
    {"chunk_target_tokens": 0},
    {"use_dense": False, "use_sparse": False},
    {"airlabs_reserve": 1000, "airlabs_monthly_quota": 1000},    # every live lookup refused
    {"min_grounding_ratio": 1.5},
])
def test_settings_reject_impossible_numbers(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **bad)


def test_derived_settings_are_revalidated():
    """model_copy(update=...) skips validation; the eval stages derive settings that way."""
    from pydantic import ValidationError

    from flight_delay.config import revalidate

    base = Settings(_env_file=None)
    assert revalidate(base.model_copy(update={"chunk_target_tokens": 256})).chunk_target_tokens == 256
    with pytest.raises(ValidationError):
        revalidate(base.model_copy(update={"final_k": 999}))


def test_settings_accept_unrelated_environment_variables(monkeypatch):
    """JUDGE_* and OS variables share the environment; they must not break startup."""
    monkeypatch.setenv("JUDGE_MODEL", "some-judge")
    monkeypatch.setenv("SOME_OS_VARIABLE", "x")
    assert Settings(_env_file=None).final_k == 8


# ==========================================================================
# Safe indexing and index identity (no database needed)
# ==========================================================================


def _identity_for(settings, **overrides):
    from flight_delay.ingest import PARSER_VERSION

    ident = {k: getattr(settings, k) for k in (
        "embedder", "embedding_model", "embedding_dim", "chunk_target_tokens",
        "chunk_overlap_pct", "chunk_breadcrumb")}
    ident["parser_version"] = PARSER_VERSION
    return ident | overrides


def test_identity_mismatch_names_every_incompatible_setting():
    from flight_delay.ingest import PARSER_VERSION
    from flight_delay.store import identity_mismatches

    s = Settings(_env_file=None)
    assert identity_mismatches(_identity_for(s), s, PARSER_VERSION) == []
    problems = identity_mismatches(
        _identity_for(s, embedding_dim=768, chunk_target_tokens=256), s, PARSER_VERSION)
    assert any("embedding_dim" in p for p in problems)
    assert any("chunk_target_tokens" in p for p in problems)
    assert identity_mismatches(_identity_for(s, parser_version="old"), s, PARSER_VERSION)
    assert identity_mismatches(None, s, PARSER_VERSION)


def test_no_corpus_operation_can_drop_conversation_tables():
    """store.drop() used to run DROP TABLE messages, conversations, chunks on every --reset."""
    import flight_delay.store as store_mod

    source = Path(store_mod.__file__).read_text(encoding="utf-8")
    assert not hasattr(store_mod.PgVectorStore, "drop")
    for stmt in re.findall(r"(?i)(?:drop|truncate)\s+table[^\n\"]*", source):
        assert "messages" not in stmt and "conversations" not in stmt, stmt
    for stmt in re.findall(r"(?i)delete\s+from\s+\w+", source):
        assert "messages" not in stmt and "conversations" not in stmt, stmt


def test_index_build_fails_before_the_database_on_a_missing_document(monkeypatch, capsys):
    import index_corpus

    monkeypatch.setattr(index_corpus, "CORPUS", index_corpus.CORPUS + [
        {"doc_id": "ghost", "path": "government_mandates/US/does-not-exist.pdf",
         "doc_title": "Ghost", "publisher": "x", "jurisdiction": "US", "doc_type": "regulation"}])

    def no_database(*a, **kw):
        raise AssertionError("the database must not be touched when the corpus is incomplete")

    import flight_delay.store as store_mod
    monkeypatch.setattr(store_mod, "PgVectorStore", no_database)
    monkeypatch.setenv("EMBEDDER", "hash")
    monkeypatch.setenv("RERANKER", "none")
    from flight_delay.config import get_settings
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit):
            index_corpus.main([])
    finally:
        get_settings.cache_clear()
    out = capsys.readouterr().out
    assert "INDEX NOT BUILT" in out and "does-not-exist.pdf" in out


def test_build_chunks_reports_parse_failures(monkeypatch, capsys):
    import index_corpus

    first = index_corpus.CORPUS[0]
    monkeypatch.setattr(index_corpus, "CORPUS", [first])

    def broken(*a, **kw):
        raise ValueError("unreadable")

    import flight_delay.ingest as ingest_mod
    monkeypatch.setattr(ingest_mod, "parse_document", broken)
    chunks, problems = index_corpus.build_chunks(Settings(_env_file=None, embedder="hash"))
    assert chunks == [] and problems and "parse failed" in problems[0]


def test_index_manifest_is_never_written_into_data(monkeypatch):
    import index_corpus

    assert not index_corpus.MANIFEST_OUT.resolve().is_relative_to(index_corpus.DATA.resolve())
    monkeypatch.setenv("EMBEDDER", "hash")
    from flight_delay.config import get_settings
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit, match="must not be inside"):
            index_corpus.main(["--manifest-out", str(index_corpus.DATA / "x.json")])
    finally:
        get_settings.cache_clear()


def test_ready_refuses_an_index_built_with_other_settings(monkeypatch):
    """Rows in the table are not enough: vectors from another model return nonsense."""
    from fastapi.testclient import TestClient

    import flight_delay.api as api

    settings = Settings(_env_file=None, embedder="hash", reranker="none")

    class IdentityStore(MemoryStore):
        identity = None

        def count(self):
            return 10

        def active_identity(self):
            return self.identity

    store = IdentityStore()
    fake = type("P", (), {"retriever": type("R", (), {"store": store})()})()
    monkeypatch.setitem(api.STATE, "pipeline", fake)
    monkeypatch.setitem(api.STATE, "settings", settings)
    client = TestClient(api.app)

    store.identity = _identity_for(settings, embedding_model="some-other-model")
    r = client.get("/ready")
    assert r.status_code == 503 and "embedding_model" in r.json()["detail"]

    store.identity = _identity_for(settings)
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["chunks"] == 10


# ==========================================================================
# Generation client, request budget, test doubles (Step 3; no network)
# ==========================================================================


def _llm(handler, **kw):
    import httpx

    from flight_delay.generation import LLMClient

    waits = []
    client = LLMClient("https://llm.test/v1", "m", "key", transport=httpx.MockTransport(handler),
                       sleep=waits.append, provider="groq", **kw)
    return client, waits


def _ok(content="Refunds are due within 7 days [S1].", finish="stop", usage=None):
    import httpx

    body = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def test_llm_retries_rate_limits_honouring_retry_after():
    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "2"}, text="slow down")
        return _ok()

    client, waits = _llm(handler, max_retries=3)
    assert client.complete("sys", "user") == "Refunds are due within 7 days [S1]."
    assert len(calls) == 3 and waits == [2.0, 2.0] and client.last.attempts == 3


def test_llm_gives_up_after_bounded_retries():
    import httpx

    from flight_delay.generation import LLMTransientError

    def handler(request):
        return httpx.Response(503, text="overloaded")

    client, waits = _llm(handler, max_retries=2, retry_max_wait_s=5)
    with pytest.raises(LLMTransientError, match="3 attempts"):
        client.complete("sys", "user")
    assert len(waits) == 2 and all(0 < w <= 5 for w in waits)


def test_llm_does_not_retry_a_rejected_request():
    import httpx

    from flight_delay.generation import LLMRequestError

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="invalid api key")

    client, waits = _llm(handler, max_retries=3)
    with pytest.raises(LLMRequestError, match="401"):
        client.complete("sys", "user")
    assert len(calls) == 1 and waits == []


# Shaped on the pilot's Groq refusal (Session 14c). The old recorder cut the real message
# right after "Requested ", so everything from there on is illustrative, not recorded.
_TOO_LARGE_MESSAGE = (
    "Request too large for model `qwen/qwen3.8-27b` in organization `org_01kzpyr3d3ea6avh7pe1kmaa3q` "
    "service tier `on_demand` on output tokens per minute (OTPM): Limit 1000, Requested 4096, please "
    "reduce your message size and try again. Need more tokens? Upgrade to Dev Tier today at "
    "https://console.groq.com/settings/billing" + " (padding to exceed the old cut)" * 12)


@pytest.mark.parametrize("status, body", [
    (429, {"error": {"message": _TOO_LARGE_MESSAGE, "type": "tokens", "code": "rate_limit_exceeded"}}),
    (413, {"error": {"message": "Request Entity Too Large", "type": "invalid_request_error",
                     "code": "request_too_large"}}),
    (429, {"error": {"message": "too big", "type": "tokens", "code": "request_too_large"}}),
])
def test_a_request_too_large_is_never_retried_and_keeps_the_whole_error(status, body):
    import httpx

    from flight_delay.generation import LLMRequestError

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"retry-after": "7"}, json=body)

    client, waits = _llm(handler, max_retries=3)
    client.api_key = "gsk_THISISASECRETKEYVALUE123"
    with pytest.raises(LLMRequestError) as info:
        client.complete("sys", "user")
    assert len(calls) == 1 and waits == []          # waiting does not shrink a request
    e = info.value
    assert (e.status, e.provider_code, e.retry_after_s) == (status, body["error"]["code"], 7.0)
    assert e.provider_type == body["error"]["type"]
    if body["error"]["message"] == _TOO_LARGE_MESSAGE:
        assert len(e.provider_message) > 600 and e.provider_message.endswith("old cut)")
        assert "Limit 1000, Requested 4096" in str(e) and "org_<redacted>" in str(e)
        assert "org_01kzpyr3d3ea6avh7pe1kmaa3q" not in str(e) + e.provider_message


def test_error_text_is_sanitized_but_never_truncated():
    from flight_delay.generation import sanitize_error_text

    text = ("HTTP 401 Invalid API Key gsk_abcdefghijklmnop Authorization: Bearer sk-live-123456789 "
            'api_key="hunter2hunter2" org_ABC123 mykeyvalue99 ' + "x" * 2000)
    out = sanitize_error_text(text, ("mykeyvalue99",))
    for secret in ("gsk_abcdefghijklmnop", "sk-live-123456789", "hunter2hunter2", "org_ABC123", "mykeyvalue99"):
        assert secret not in out
    assert out.endswith("x" * 2000)


def test_retry_after_is_honoured_in_full_and_a_longer_one_is_not_retried_early():
    import httpx

    from flight_delay.generation import LLMTransientError

    calls = []

    def slow_then_ok(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "45"},
                                  json={"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}})
        return _ok()

    client, waits = _llm(slow_then_ok, max_retries=3, retry_max_wait_s=120)
    completion = client.generate("sys", "user")
    assert waits == [45.0]                             # not cut to the 30 s default
    [event] = completion.http_errors
    assert event["status"] == 429 and event["retry_after_s"] == 45.0 and event["waited_s"] == 45.0
    assert event["provider_code"] == "rate_limit_exceeded" and event["provider_message"] == "Rate limit reached"

    calls.clear()
    client, waits = _llm(slow_then_ok, max_retries=3, retry_max_wait_s=30)
    with pytest.raises(LLMTransientError, match="Retry-After") as info:
        client.complete("sys", "user")
    assert len(calls) == 1 and waits == [] and info.value.retry_after_s == 45.0


def test_calls_are_paced_by_the_minimum_interval():
    import httpx

    from flight_delay.generation import LLMClient

    now = [1000.0]
    waits = []

    def sleep(s):
        waits.append(round(s, 3))
        now[0] += s

    def handler(request):
        now[0] += 5.0                                  # each request takes 5 s
        return _ok()

    client = LLMClient("https://llm.test/v1", "m", "key", transport=httpx.MockTransport(handler),
                       provider="groq", sleep=sleep, clock=lambda: now[0], min_call_interval_s=60)
    client.complete("sys", "user")
    assert waits == []                                 # the first call is not delayed
    client.complete("sys", "user")
    assert waits == [55.0]                             # 60 s between request starts
    now[0] += 100
    client.complete("sys", "user")
    assert waits == [55.0]                             # already long enough ago


def test_llm_retries_timeouts_and_dropped_connections():
    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        if len(calls) == 2:
            raise httpx.ConnectError("reset", request=request)
        return _ok()

    client, _ = _llm(handler, max_retries=3)
    assert "[S1]" in client.complete("sys", "user") and len(calls) == 3


@pytest.mark.parametrize("response_kwargs, match", [
    ({"finish": "length"}, "truncated"),
    ({"content": None}, "no answer text"),
    ({"content": "   "}, "no answer text"),
])
def test_llm_never_returns_a_truncated_or_empty_answer(response_kwargs, match):
    from flight_delay.generation import LLMResponseError

    client, _ = _llm(lambda request: _ok(**response_kwargs))
    with pytest.raises(LLMResponseError, match=match):
        client.complete("sys", "user")


def test_llm_rejects_a_malformed_body():
    import httpx

    from flight_delay.generation import LLMResponseError

    client, _ = _llm(lambda request: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(LLMResponseError, match="unreadable"):
        client.complete("sys", "user")


def test_llm_strips_reasoning_and_sends_extra_body():
    import json

    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _ok(content="<think>the user wants rules</think>\nYou are owed a refund [S1].")

    client, _ = _llm(handler, extra_body={"reasoning_format": "hidden"})
    assert client.complete("sys", "user") == "You are owed a refund [S1]."
    assert seen["reasoning_format"] == "hidden" and seen["model"] == "m" and seen["temperature"] == 0.0


def test_llm_usage_feeds_token_and_cost_metrics_only_when_available():
    from prometheus_client import REGISTRY

    def sample(name, **labels):
        return REGISTRY.get_sample_value(name, labels) or 0.0

    before_prompt = sample("rag_tokens_total", kind="prompt")
    before_cost = sample("rag_query_cost_usd_count")
    usage = {"prompt_tokens": 1000, "completion_tokens": 100}

    client, _ = _llm(lambda request: _ok(usage=usage))
    client.complete("sys", "user")
    assert sample("rag_tokens_total", kind="prompt") == before_prompt + 1000
    assert sample("rag_query_cost_usd_count") == before_cost      # no prices: cost unavailable

    priced, _ = _llm(lambda request: _ok(usage=usage),
                     price_input_per_mtok=0.5, price_output_per_mtok=1.0)
    priced.complete("sys", "user")
    assert sample("rag_query_cost_usd_count") == before_cost + 1


def test_build_llm_never_falls_back_to_a_stand_in():
    from flight_delay.generation import EchoLLM, LLMClient, build_llm

    with pytest.raises(RuntimeError, match="test double"):
        build_llm(Settings(_env_file=None, llm_provider="echo"))
    with pytest.raises(RuntimeError, match="LLM_BASE_URL is empty"):
        build_llm(Settings(_env_file=None, llm_base_url=""))
    assert isinstance(build_llm(Settings(_env_file=None, llm_provider="echo",
                                         allow_test_doubles=True)), EchoLLM)
    groq = build_llm(Settings(_env_file=None, llm_provider="groq", llm_api_key="k",
                              llm_base_url="https://api.groq.com/openai/v1", llm_model="some-model"))
    assert isinstance(groq, LLMClient) and groq.provider == "groq"


def test_groq_provider_requires_a_key():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="LLM_API_KEY"):
        Settings(_env_file=None, llm_provider="groq")


def test_evidence_budget_fits_the_whole_request_in_the_context_window():
    from flight_delay.generation import SYSTEM_PROMPT, evidence_token_budget

    question = "What am I owed after my flight was cancelled?"
    small = Settings(_env_file=None, llm_context_window=8192)
    budget = evidence_token_budget(small, question, None)
    fixed = len(SYSTEM_PROMPT) // 4 + len(question) // 4
    assert 0 < budget < small.context_token_budget
    assert fixed + budget + small.llm_max_tokens <= small.llm_context_window
    # history and retry notes take their share
    history = [("user", "x" * 400), ("assistant", "y" * 400)] * 2
    assert evidence_token_budget(small, question, history) < budget
    # a large window is capped by context_token_budget, not by the window
    big = Settings(_env_file=None, llm_context_window=131072)
    assert evidence_token_budget(big, question, None) == big.context_token_budget


def test_serving_refuses_test_doubles_and_a_window_too_small_for_the_prompt():
    from flight_delay.api import check_serving_settings

    # there is no default generator: the defaults alone are refused
    with pytest.raises(RuntimeError, match="no generator is configured"):
        check_serving_settings(Settings(_env_file=None))
    groq = {"llm_provider": "groq", "llm_base_url": "https://api.groq.com/openai/v1",
            "llm_model": "some-model", "llm_api_key": "k"}
    check_serving_settings(Settings(_env_file=None, **groq))   # a configured generator is fine
    for bad, match in (
        ({"embedder": "hash"}, "EMBEDDER=hash"),
        ({"llm_provider": "echo"}, "echo generator"),
        ({"use_flight_fixtures": True}, "USE_FLIGHT_FIXTURES"),
        ({"llm_context_window": 4096}, "LLM_CONTEXT_WINDOW"),
    ):
        with pytest.raises(RuntimeError, match=match):
            check_serving_settings(Settings(_env_file=None, **(groq | bad)))
    check_serving_settings(Settings(_env_file=None, embedder="hash", allow_test_doubles=True, **groq))


def test_pipeline_request_fits_the_configured_window(monkeypatch):
    """The assembled prompt must fit llm_context_window, not just context_token_budget."""
    from flight_delay.generation import SYSTEM_PROMPT
    from flight_delay.pipeline import RagPipeline

    seen = {}

    class Recorder:
        def complete(self, system, user):
            seen["chars"] = len(system) + len(user)
            return "You are owed a refund [S1]."

    big_chunks = [Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", doc_title="T", publisher="p",
                        jurisdiction="US", airline_iata="", doc_type="regulation",
                        section_id="s", breadcrumb="b", text="refund " * 1400,
                        embed_text="refund", source_url="", token_estimate=1400)
                  for i in range(6)]

    class Retriever:
        def retrieve(self, query, *, top_k=None, candidates_k=None, flt=None, exclude_topics=(), lanes=(), min_airline=None):
            from flight_delay.retrieval import RetrievalResult

            return RetrievalResult(chunks=big_chunks, timings_ms={}, n_dense=6, n_sparse=6,
                                   n_fused=6)

    settings = Settings(_env_file=None, embedder="hash", reranker="none",
                        confidence_gate_enabled=False, llm_context_window=8192)
    pipe = RagPipeline(Retriever(), Recorder(), _FakeTool({}), settings, store=_FakeStore())
    pipe.run("Can I get a refund if American cancels my domestic flight from Chicago to Denver?")
    assert seen["chars"] // 4 + settings.llm_max_tokens <= settings.llm_context_window
    assert len(SYSTEM_PROMPT) // 4 < settings.llm_context_window


def test_indexer_refuses_a_hash_embedder_index_without_opt_in(monkeypatch, tmp_path):
    import index_corpus

    from flight_delay.config import get_settings

    monkeypatch.setenv("EMBEDDER", "hash")
    monkeypatch.delenv("ALLOW_TEST_DOUBLES", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit, match="test-double index"):
            index_corpus.main(["--manifest-out", str(tmp_path / "m.json")])
    finally:
        get_settings.cache_clear()


# ==========================================================================
# Step 4: runtime correctness (no network, no database)
# ==========================================================================


class _StateStore(_FakeStore):
    """A conversation store that also keeps the per-conversation route state."""

    def __init__(self):
        super().__init__()
        self.states = {}

    def load_state(self, conv_id):
        return self.states.get(conv_id)

    def save_state(self, conv_id, state):
        import json

        self.states[conv_id] = json.loads(json.dumps(state))   # must survive JSON


class _RaisingLLM:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        raise self.exc


def _run_pipeline(llm, retriever=None, store=None, **settings):
    from flight_delay.pipeline import RagPipeline

    s = Settings(_env_file=None, embedder="hash", reranker="none",
                 **({"confidence_gate_enabled": False} | settings))
    return RagPipeline(retriever or _OneChunkRetriever(), llm, _FakeTool({}), s,
                       store=store if store is not None else _FakeStore())


def test_an_answer_that_still_fails_validation_is_never_shown():
    from flight_delay.pipeline import VALIDATION_FAILURE_MESSAGE

    invented = "You are owed $1,550 in cash within 3 days for any cancellation."
    store = _FakeStore()
    pipe = _run_pipeline(_ScriptedLLM(invented, invented), store=store)
    a = pipe.run("Can I get a refund if American cancels my domestic flight?", conversation_id="c1")
    assert a.outcome == "validation_failed" and a.generation_attempted
    assert VALIDATION_FAILURE_MESSAGE in a.text and "1,550" not in a.text
    assert a.citations == [] and a.diagnostics["rejected_answer"] == invented
    assert all("1,550" not in content for _, content in store.msgs)   # not saved either


def test_an_answer_failing_only_on_a_bullet_is_shown_without_it():
    """Session 37: every attempt failed on one uncited advice bullet; the cited rest
    is shown, the bullet is gone from the text AND from what follow-ups may quote."""
    answer = ("By law you are owed a refund [S1].\n\n"
              "- Ask American for the refund form.\n"
              "- If you prefer cash, request the $1,550 compensation.")
    store = _FakeStore()
    pipe = _run_pipeline(_ScriptedLLM(answer, answer), store=store)
    a = pipe.run("Can I get a refund if American cancels my domestic flight?", conversation_id="c1")
    assert a.outcome == "answered" and a.retry_count == 1
    assert "1,550" not in a.text and "refund form" in a.text
    assert a.diagnostics["dropped_uncited_list_items"] == [
        "- If you prefer cash, request the $1,550 compensation."]
    assert "1,550" not in a.diagnostics["model_answer"]
    assert all("1,550" not in content for _, content in store.msgs)


def test_validation_failure_without_retry_is_also_replaced():
    invented = "You are owed $1,550 in cash within 3 days for any cancellation."
    pipe = _run_pipeline(_ScriptedLLM(invented))
    a = pipe.run("Can I get a refund if American cancels my domestic flight?", allow_retry=False)
    assert a.outcome == "validation_failed" and "1,550" not in a.text


def test_api_response_never_carries_the_rejected_answer(api_client):
    invented = "You are owed $1,550 in cash within 3 days for any cancellation."
    pipe = _run_pipeline(_ScriptedLLM(invented, invented))
    r = api_client(pipe).post("/ask", json={"question": "Can I get a refund if American cancels my domestic flight?"})
    assert r.status_code == 200 and "1,550" not in r.text
    assert r.json()["outcome"] == "validation_failed"


@pytest.mark.parametrize("exc_name", ["LLMTransientError", "LLMRequestError", "LLMResponseError"])
def test_a_model_failure_returns_a_safe_message_not_a_500(api_client, exc_name):
    import flight_delay.generation as gen
    from flight_delay.pipeline import LLM_ERROR_MESSAGE

    llm = _RaisingLLM(getattr(gen, exc_name)("boom"))
    pipe = _run_pipeline(llm, store=_FakeStore())
    r = api_client(pipe).post("/ask", json={"question": "Can I get a refund if American cancels my domestic flight?"})
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "llm_error" and LLM_ERROR_MESSAGE in body["answer"]
    assert body["citations"] == [] and "boom" not in body["answer"]


def test_gated_turns_are_saved_and_labelled_with_a_bounded_reason():
    from prometheus_client import REGISTRY

    store = _FakeStore()
    before = REGISTRY.get_sample_value("rag_gate_abstentions_total", {"reason": "no_results"}) or 0
    pipe = _run_pipeline(_ScriptedLLM(), retriever=_FakeRetriever(), store=store,
                         confidence_gate_enabled=True)
    a = pipe.run("How much compensation for denied boarding?", conversation_id="c1")
    assert a.outcome == "gated" and not a.generation_attempted
    assert [role for role, _ in store.msgs] == ["user", "assistant"]
    assert REGISTRY.get_sample_value("rag_gate_abstentions_total", {"reason": "no_results"}) == before + 1


def test_outcomes_are_structured_for_clarify_and_answer():
    pipe, _ = _pipeline()
    assert pipe.run("my delta flight was delayed 5 hours", conversation_id="c1").outcome == "clarified"
    answered = _run_pipeline(_ScriptedLLM("A refund is owed when the airline cancels [S1]."))
    a = answered.run("Can I get a refund if American cancels my domestic flight?")
    assert a.outcome == "answered" and a.built_context is not None
    assert a.prompt_sha256 and a.built_context.used_chunks == a.context_chunks


def test_follow_up_keeps_the_route_and_carrier_it_settled():
    store = _StateStore()
    retriever = _OneChunkRetriever()
    llm = _ScriptedLLM(*["A refund is owed when the airline cancels [S1]."] * 4)
    pipe = _run_pipeline(llm, retriever=retriever, store=store)

    first = pipe.run("My Delta flight from Paris to Atlanta was cancelled. What am I owed?",
                     conversation_id="c1")
    assert first.outcome == "answered"
    first_flt = retriever.calls[-1][1]
    assert first_flt["airline_scope"] == "DL" and "EU" in first_flt["jurisdiction_governing"]

    # An ordinary follow-up names nothing: same carrier, same regimes, no question
    # back. allow_followup=False because "What about a hotel?" is one of the moves
    # followups.py answers from the previous turn, and a template reply retrieves
    # nothing for this assertion to read.
    hotel = pipe.run("What about a hotel?", conversation_id="c1", allow_followup=False)
    assert hotel.outcome == "answered" and not hotel.clarification
    assert retriever.calls[-1][1]["airline_scope"] == "DL"
    assert retriever.calls[-1][1]["jurisdiction_governing"] == first_flt["jurisdiction_governing"]

    # The template path settles the same carrier and regimes for the turn after it.
    templated = pipe.run("What about a hotel?", conversation_id="c1")
    assert templated.outcome == "followup"
    assert store.states["c1"]["airline"] == "DL"
    assert store.states["c1"]["flt"]["jurisdiction_governing"] == first_flt["jurisdiction_governing"]

    # A personal disruption follow-up would normally ask for the route; it does not.
    delayed = pipe.run("And if my flight had only been delayed 4 hours?", conversation_id="c1")
    assert not delayed.clarification
    assert retriever.calls[-1][1]["jurisdiction_governing"] == first_flt["jurisdiction_governing"]

    # Naming a new route re-routes, keeping the carrier.
    reverse = pipe.run("What if I had flown from Atlanta to Paris instead?", conversation_id="c1")
    assert not reverse.clarification
    assert retriever.calls[-1][1]["airline_scope"] == "DL"
    assert retriever.calls[-1][1]["jurisdiction_governing"] == ["US"]


def test_a_partial_flight_record_does_not_skip_the_route_question():
    partial = _flight(None, "CDG")
    pipe, retriever = _pipeline({"UA57": partial})
    a = pipe.run("My flight UA57 was cancelled, what am I owed?", conversation_id="c1")
    assert a.outcome == "clarified" and "I found flight UA57" in a.text
    assert "What is your flight number" not in a.text
    assert retriever.calls == []


def test_flight_data_for_another_day_is_used_only_for_the_airports():
    from datetime import date

    from flight_delay.pipeline import plan_turn

    record = _flight("CDG", "EWR")
    record.dep_time = "2026-09-10 09:00"
    record.dep_delayed = 240

    yesterday = plan_turn("My flight UA57 yesterday was delayed, what am I owed?", [],
                          lambda n: record, today=date(2026, 9, 16))
    assert yesterday.live_status_usable is False and "2026-09-10" in yesterday.flight_note
    assert "EU" in yesterday.flt["jurisdiction_governing"]          # airports still route it

    same_day = plan_turn("My flight UA57 on September 10 was delayed, what am I owed?", [],
                         lambda n: record, today=date(2026, 9, 16))
    assert same_day.live_status_usable is True
    assert same_day.flight_note == "I found UA57: CDG to EWR, cancelled. If that's not your flight, tell me."
    assert yesterday.flight_note.startswith("I found UA57: CDG to EWR. ")  # another day: airports only

    no_date = plan_turn("My flight UA57 was delayed, what am I owed?", [], lambda n: record,
                        today=date(2026, 9, 16))
    assert no_date.live_status_usable is True


def test_pipeline_withholds_another_days_status_from_the_model_and_the_passenger():
    record = _flight("CDG", "EWR")
    record.dep_time = "2026-09-10 09:00"
    record.dep_delayed = 240
    llm = _ScriptedLLM("A refund is owed when the airline cancels [S1].")
    from flight_delay.pipeline import RagPipeline

    s = Settings(_env_file=None, embedder="hash", reranker="none", confidence_gate_enabled=False)
    pipe = RagPipeline(_OneChunkRetriever(), llm, _FakeTool({"UA57": record}), s, store=_FakeStore())
    a = pipe.run("My flight UA57 last week was delayed, what am I owed?")
    assert a.live_data is None and "240" not in llm.prompts[0]
    assert "only to identify the airports" in a.text


def test_codeshare_is_flagged_to_the_passenger_and_the_model():
    from flight_delay.generation import flight_block
    from flight_delay.pipeline import plan_turn

    record = _flight("JFK", "LHR")
    record.operating_airline_iata = "BA"
    record.operating_flight_iata = "BA1520"
    plan = plan_turn("My flight UA57 was delayed, what am I owed?", [], lambda n: record)
    assert "operated by British Airways" in plan.flight_note
    assert "Operated by: BA as BA1520 (codeshare)" in flight_block(record)


def test_recorded_and_synthetic_flight_data_are_labelled_as_not_current():
    from flight_delay.generation import flight_block

    record = _flight("JFK", "LHR")
    record.source = "fixture"
    assert "RECORDED" in flight_block(record) and "NOT current" in flight_block(record)
    record.source = "synthetic"
    assert "SYNTHETIC" in flight_block(record)


# -- validator ----------------------------------------------------------------


def _ctx_with_flight(delay=95):
    live = _flight("ORD", "DFW")
    live.dep_delayed = delay
    return build_context([_chunk("c1", "Refunds are due for cancelled flights.")], live)


def test_an_abstention_phrase_no_longer_excuses_other_uncited_claims():
    ctx = _ctx_with_flight()
    mixed = ("The sources do not specify the meal policy. "
             "You are entitled to $750 in compensation for this delay.")
    report = validate_answer(mixed, ctx)
    assert not report.ok and len(report.uncited_factual_sentences) == 1
    same_sentence = "The sources do not specify this, but you are entitled to $750 in compensation."
    assert not validate_answer(same_sentence, ctx).ok
    pure = "The provided sources do not cover this question."
    assert validate_answer(pure, ctx).ok and validate_answer(pure, ctx).is_abstention


def test_restating_the_flight_data_needs_no_marker_but_new_numbers_do():
    ctx = _ctx_with_flight(95)
    ok = validate_answer("Flight UA57 is delayed by 95 minutes. Refunds are due [S1].", ctx)
    assert ok.ok and ok.flight_data_sentences == 1
    wrong = validate_answer("Flight UA57 is delayed by 195 minutes. Refunds are due [S1].", ctx)
    assert not wrong.ok
    money = validate_answer("Flight UA57 is delayed by 95 minutes, so you are owed a refund. "
                            "Refunds are due [S1].", ctx)
    assert not money.ok      # a rights claim is never excused as flight data


def test_marker_validity_and_citation_coverage_are_reported_separately():
    ctx = _ctx_with_flight()
    r = validate_answer("Refunds are due within 7 days [S1]. You must ask within 30 days.", ctx)
    assert r.unknown_markers == [] and r.citation_coverage == 0.5
    assert validate_answer("Nothing to see here, thanks for asking!", ctx).citation_coverage is None


# -- validator: the second-pilot regressions (actual groq-gpt-oss-20b answers, Session 14c;
#    evals/runs/generation/20260917T031903.json, final attempt of each case) -------------

_PILOT_UK_US_01_QUESTION = ("My United flight from LHR to EWR was delayed 4 hours due to a mechanical "
                            "issue. What am I owed?")
_PILOT_UK_US_01_ANSWER = (
    '**1. Flight situation**  \nYour United flight from London Heathrow (LHR) to Newark‑EWR was delayed '
    '4 hours because of a mechanical issue.  \n\n**2. Applicable law (Tier 1 – UK Regulation '
    '261)**  \nBecause the flight departed from the UK, UK 261 applies.  \n- Article 6 requires the '
    'airline to offer assistance (meals, transport, accommodation) when a flight is delayed 4 hours or '
    'more on a route over 1 500 km [S3].  \n- Article 7 entitles passengers to a fixed cash '
    'compensation when the delay to the final destination is 3 hours or more, unless the airline can '
    'prove extraordinary circumstances. The amount is set by UK law but is not listed in the sources you '
    'provided [S3].  \n\n**3. Airline’s own policy (Tier 2 – United Customer Commitment)**  '
    '\nUnited’s commitment adds the following:  \n- Rebooking on the next available United flight in the '
    'same cabin at no extra cost, or on a partner flight if no United seat is available [S6].  \n- A '
    'digital or printed meal voucher for a reasonable‑cost meal if the delay is more than three hours '
    'and caused by something within United’s control [S6].  \n- A digital or printed hotel voucher '
    'for an overnight stay if the delay keeps you away from home and a partner hotel room is '
    'unavailable [S6].  \n- A voucher for reasonable ground‑transportation costs to the hotel if no '
    'shuttle is provided [S6].  \n- If you decline any of these vouchers, United will not reimburse you '
    'for independently incurred expenses [S6].  \n\n**4. Cause caveat**  \nBecause the delay was caused '
    'by a mechanical issue (an event within United’s control), the airline’s policy applies fully. If '
    'the cause were something outside United’s control, the airline would still be required to rebook you '
    'but would not be obligated to provide the meal, hotel, or transport vouchers [S6].  \n\n**5. '
    'Actionable next steps**  \n1. Contact United (via the app, website, or airport kiosk) to confirm your '
    'rebooking and request the meal, hotel, and transport vouchers.  \n2. Keep receipts for any '
    'independently paid meals or lodging; you may submit them for reimbursement if United offers a voucher '
    'but you decline it.  \n3. If you wish to claim the cash compensation under UK 261, file a claim '
    'with United’s customer‑service department and provide the flight details and delay evidence.  '
    '\n\n*All rights above are based on the sources provided.*')
_PILOT_TRAP_04_QUESTION = ("How much compensation do Canada's Air Passenger Protection Regulations give for a "
                           "6-hour delay on my United flight from Toronto? Sorry, I don't have my flight "
                           "number or the airport details.")
_PILOT_TRAP_04_ANSWER = (
    'I’m sorry, but none of the sources provided contain information about Canada’s Air Passenger '
    'Protection Regulations or the compensation they provide for a 6‑hour delay on a United flight '
    'departing from Toronto. To find the exact amount you are entitled to, please consult the Canadian '
    'Transportation Agency’s official guidance or contact United’s customer service directly.')


def _six_sources_no_flight():
    return build_context([_chunk(f"c{i}", f"Clause {i} text.") for i in range(1, 7)], None)


def test_pilot_uk_us_01_flags_only_its_uncited_legal_claims():
    """The old validator flagged five sentences. Classified before the change:
      1. "Your United flight from London Heathrow (LHR) to Newark-EWR was delayed 4 hours
         because of a mechanical issue."                       -> user-provided fact
      2. "Because the flight departed from the UK, UK 261 applies." (after the heading
         "Applicable law (Tier 1 - UK Regulation 261)")        -> policy (legal) claim
      3. "Article 7 entitles passengers to a fixed cash compensation when the delay ... is
         3 hours or more, unless ... extraordinary circumstances." -> policy (legal) claim
      4. "Actionable next steps** 1." (heading + list number)  -> recommendation (no claim)
      5. "If you wish to claim the cash compensation under UK 261, file a claim with
         United's customer-service department ..."             -> recommendation
    Only 2 and 3 still need a marker, so the answer is still (correctly) rejected."""
    ctx = _six_sources_no_flight()
    report = validate_answer(_PILOT_UK_US_01_ANSWER, ctx, question=_PILOT_UK_US_01_QUESTION)
    assert not report.ok
    assert report.failures == ["2 factual sentence(s) carry no citation"]
    assert report.uncited_factual_sentences == [
        "Because the flight departed from the UK, UK 261 applies.",
        "Article 7 entitles passengers to a fixed cash compensation when the delay to the final "
        "destination is 3 hours or more, unless the airline can prove extraordinary circumstances.",
    ]
    assert report.user_fact_sentences == 1 and report.recommendation_sentences == 1
    assert not any("next steps" in s or "Flight situation" in s for s in report.uncited_factual_sentences)
    # the restated situation is excused only because the PASSENGER said it
    without_question = validate_answer(_PILOT_UK_US_01_ANSWER, ctx)
    assert any(s.startswith("Your United flight") for s in without_question.uncited_factual_sentences)


def test_pilot_trap_04_source_coverage_decline_is_accepted():
    """The correct decline the old validator rejected ("2 factual sentence(s) carry no
    citation" + "no citations at all and not an explicit abstention")."""
    report = validate_answer(_PILOT_TRAP_04_ANSWER, _six_sources_no_flight(), question=_PILOT_TRAP_04_QUESTION)
    assert report.ok and report.is_abstention, report.failures
    assert report.abstention_sentences == 1 and report.recommendation_sentences == 1
    assert report.uncited_factual_sentences == [] and report.citations == []


def test_a_decline_does_not_excuse_a_claim_anywhere_in_the_answer():
    ctx, q = _six_sources_no_flight(), _PILOT_TRAP_04_QUESTION
    for extra in ("You are entitled to CAD 400 for this delay.",
                  "United must still pay you compensation for a 6-hour delay.",
                  "Contact United within 30 days to claim compensation."):
        report = validate_answer(_PILOT_TRAP_04_ANSWER + " " + extra, ctx, question=q)
        assert not report.ok and not report.is_abstention, extra
        assert report.uncited_factual_sentences == [extra]
    # inside the decline sentence itself: a joined claim, an amount or an invented number
    for sentence in ("None of the sources provided cover Canada, but you are entitled to compensation.",
                     "The sources do not cover the CAD 400 you are owed for this delay.",
                     "The sources do not say whether the 72-hour rule applies to your delay."):
        assert not validate_answer(sentence, ctx, question=q).ok, sentence
    # a policy statement is not a source-coverage decline, even without an amount
    assert not validate_answer("United does not cover hotel costs for weather delays.", ctx, question=q).ok


def test_user_facts_recommendations_and_claims_are_told_apart():
    ctx, q = _six_sources_no_flight(), _PILOT_UK_US_01_QUESTION
    lead = "Article 6 requires meals when a flight is delayed 4 hours [S3]. "
    accepted = (
        "Your United flight from LHR to EWR was delayed 4 hours due to a mechanical issue.",
        "If United refuses, file a complaint with the US Department of Transportation about the compensation.",
        "You may want to contact the UK Civil Aviation Authority if your UK261 compensation claim is refused.",
        "Please keep your boarding pass and receipts for 4 hours of expenses.",
    )
    for sentence in accepted:
        assert validate_answer(lead + sentence, ctx, question=q).ok, sentence
    flagged = (
        "Your United flight from LHR to EWR was delayed 5 hours due to a mechanical issue.",  # new number
        "Your United flight from LHR to EWR was delayed 4 hours, so United will rebook you.",  # policy
        "Ask United for the hotel voucher they must provide after 4 hours.",                  # obligation
        "File your claim with United within 4 hours of landing.",                             # deadline
        "Claim €600 from United's customer service for a 4-hour delay.",                      # amount
        "Contact United, which is liable for delays of 4 hours or more.",                     # claim in advice
    )
    for sentence in flagged:
        report = validate_answer(lead + sentence, ctx, question=q)
        assert not report.ok and report.uncited_factual_sentences == [sentence], sentence


def test_markdown_layout_is_not_validated_as_sentences():
    ctx = _six_sources_no_flight()
    answer = ("## Your rights under UK Regulation 261\n\n"
              "**1. What the law says**\n"
              "- The airline must offer meals after 2 hours [S1].\n"
              "The carrier shall also offer refreshments in\n"
              "reasonable relation to the waiting time [S2].\n\n"
              "**You are owed €600 in 2 cases.**")
    report = validate_answer(answer, ctx)
    # headings are labels; a wrapped sentence is one sentence; a bold claim is still a claim
    assert report.uncited_factual_sentences == ["You are owed €600 in 2 cases."]
    assert report.cited_factual_sentences == 2


# -- API ------------------------------------------------------------------------


def test_manual_jurisdiction_scopes_regulations_without_dropping_airline_documents(api_client):
    retriever = _OneChunkRetriever()
    pipe = _run_pipeline(_ScriptedLLM("A refund is owed when the airline cancels [S1]."),
                         retriever=retriever)
    client = api_client(pipe)
    r = client.post("/ask", json={"question": "Can I get a refund if American cancels?",
                                  "jurisdiction": "EU"})
    assert r.status_code == 200
    flt = retriever.calls[-1][1]
    assert "jurisdiction" not in flt
    assert flt["jurisdiction_scope"] == ["EU"] and flt["jurisdiction_governing"] == ["EU"]
    assert client.post("/ask", json={"question": "Refund rules?", "jurisdiction": "FR"}).status_code == 422


def test_conversations_can_only_be_continued_with_their_token(api_client):
    pipe = _run_pipeline(_ScriptedLLM(*["A refund is owed when the airline cancels [S1]."] * 3))
    client = api_client(pipe, conversation_secret="test-secret")
    first = client.post("/ask", json={"question": "Can I get a refund if American cancels?"}).json()
    cid, token = first["conversation_id"], first["conversation_token"]
    assert token

    ok = client.post("/ask", json={"question": "What about a hotel?", "conversation_id": cid,
                                   "conversation_token": token})
    assert ok.status_code == 200 and ok.json()["conversation_id"] == cid
    assert client.post("/ask", json={"question": "What about a hotel?",
                                     "conversation_id": cid}).status_code == 403
    assert client.post("/ask", json={"question": "What about a hotel?", "conversation_id": cid,
                                     "conversation_token": "0" * 40}).status_code == 403
    assert client.post("/ask", json={"question": "What about a hotel?",
                                     "conversation_id": "someone-elses-id",
                                     "conversation_token": token}).status_code == 403


def test_ui_inserts_server_values_as_text_only():
    from flight_delay.api import HTML_PAGE

    assert "esc(shown)" in HTML_PAGE and "esc(c.doc_title)" in HTML_PAGE
    assert "esc(detail)" in HTML_PAGE and "safeUrl(c.source_url)" in HTML_PAGE
    assert 'rel="noopener noreferrer"' in HTML_PAGE
    assert "d.validation_failures.join" not in HTML_PAGE
    assert "+d.flight.flight+" not in HTML_PAGE and "+(d.detail||" not in HTML_PAGE


def test_in_process_rate_limit_counts_per_client_and_prunes(monkeypatch):
    from flight_delay import api

    api._BUCKET.clear()
    s = Settings(_env_file=None, rate_limit_per_minute=2)
    assert [api._rate_limited("a", s) for _ in range(3)] == [False, False, True]
    assert api._rate_limited("b", s) is False
    monkeypatch.setattr(api, "_MAX_TRACKED_CLIENTS", 5)
    monkeypatch.setattr(api.time, "time", lambda: 10**10)       # "a" and "b" have expired
    for i in range(10):                                          # a flood of new clients
        api._rate_limited(f"client-{i}", s)
    assert len(api._BUCKET) <= 5 and "a" not in api._BUCKET and "client-9" in api._BUCKET
    api._BUCKET.clear()


def test_forwarded_for_is_used_only_when_trusted():
    from types import SimpleNamespace

    from flight_delay.api import client_key

    request = SimpleNamespace(headers={"x-forwarded-for": "203.0.113.9, 10.0.0.1"},
                              client=SimpleNamespace(host="10.0.0.1"))
    assert client_key(request, Settings(_env_file=None)) == "10.0.0.1"
    assert client_key(request, Settings(_env_file=None, trust_forwarded_for=True)) == "203.0.113.9"


# -- quota, circuit breaker, cache ------------------------------------------------


def test_file_quota_reservation_is_atomic_across_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    c = FileQuotaCounter(str(tmp_path / "q.json"), monthly_quota=50)
    with ThreadPoolExecutor(max_workers=8) as pool:
        granted = sum(pool.map(lambda _: c.try_reserve(10), range(100)))
    assert granted == 40 and c.used() == 40          # never beyond quota minus reserve


def test_store_quota_fails_closed_when_the_database_is_unavailable():
    from flight_delay.tools import QuotaExceeded, QuotaUnavailable, StoreQuotaCounter

    class DownStore:
        def quota_try_reserve(self, period, limit):
            raise ConnectionError("db down")

        def quota_used(self, period):
            raise ConnectionError("db down")

    c = StoreQuotaCounter(DownStore(), 1000)
    with pytest.raises(QuotaUnavailable):
        c.try_reserve(100)
    assert issubclass(QuotaUnavailable, QuotaExceeded)   # the pipeline treats it as blocked


def test_failed_live_requests_still_count_and_error_payloads_open_the_breaker(airlabs, monkeypatch):
    from flight_delay import tools

    class ErrorPayload:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            class R:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"error": {"message": "bad"}}
            return R()

    monkeypatch.setattr(tools.httpx, "Client", ErrorPayload)
    tool = airlabs(use_flight_fixtures=False)
    for _ in range(3):
        assert tool.get_flight("AA2359") is None
    assert tool.quota.used() == 3                       # every attempt was reserved
    assert tool._open_until > 0                         # error payloads count as failures
    assert tool.get_flight("AA2359") is None and tool.quota.used() == 3   # circuit open: no call


def test_cached_flights_are_returned_as_copies(airlabs):
    tool = airlabs(use_flight_fixtures=False)
    first = tool.get_flight("AA2359")
    second = tool.get_flight("AA2359")
    assert first.source == "live" and second.source == "cache"
    second.dep_iata = "XXX"
    assert tool.get_flight("AA2359").dep_iata == "JFK"


def test_fixture_flights_are_labelled_as_fixtures(airlabs):
    fs = airlabs(use_flight_fixtures=True).get_flight("AA2359")
    assert fs.source == "fixture"


def test_unused_flight_helpers_that_wrote_into_data_are_gone():
    from flight_delay.tools import AirLabsTool

    assert not hasattr(AirLabsTool, "record_fixture") and not hasattr(AirLabsTool, "get_delays")


def test_reranker_scores_are_raw_logits(monkeypatch):
    """Sigmoid would put every score in (0, 1), where a -2.0 threshold can never fire."""
    import sys
    import types

    import torch

    seen = {}

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            seen.update(kwargs)

    fake = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    from flight_delay.retrieval import CrossEncoderReranker

    _ = CrossEncoderReranker("any-model").model   # loaded lazily on first use
    assert isinstance(seen["activation_fn"], torch.nn.Identity)
    assert CrossEncoderReranker.score_activation.startswith("identity")


def test_calibration_helper_that_bypassed_routing_is_retired():
    import flight_delay.confidence as confidence

    assert not hasattr(confidence, "calibrate_thresholds")


# ==========================================================================
# Session 12: parser fidelity, chunk size contract, gold passages, Stage 1
# ==========================================================================

_PARSED_CORPUS: dict = {}


def _corpus_sections() -> dict:
    """The real manifest parsed once (data/ is read, never written)."""
    if not _PARSED_CORPUS:
        import contextlib
        import io

        import index_corpus

        with contextlib.redirect_stdout(io.StringIO()):
            parsed, problems = index_corpus.parse_corpus()
        assert not problems, problems
        _PARSED_CORPUS.update(parsed)
    return _PARSED_CORPUS


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_every_document_has_unique_section_ids():
    for doc, sections in _corpus_sections().items():
        ids = [s.section_id for s in sections]
        assert len(ids) == len(set(ids)), f"{doc}: duplicate section ids"


def test_unique_section_ids_suffixes_repeats():
    from dataclasses import replace

    from flight_delay.ingest import unique_section_ids

    sec = _sections()[0]
    out = unique_section_ids([sec, replace(sec, text="other"), replace(sec, text="third")])
    assert [s.section_id for s in out] == [sec.section_id, f"{sec.section_id}-2",
                                           f"{sec.section_id}-3"]


def test_contract_rules_are_the_sections_not_their_numbered_list_items():
    """United's 'Rule N' headings used to go undetected, so numbered list items ('1. The
    first flight on which space is available; or') became sections spanning several rules."""
    corpus = _corpus_sections()
    ua = [s.section_id for s in corpus["ua-contract-of-carriage"]]
    assert "rule-24-flight-delays-cancellations-aircraft-changes" in ua
    assert "rule-25-denied-boarding-compensation" in ua
    assert not [i for i in ua if re.match(r"\d", i)], "a numbered list item became a section"
    dl = {s.section_id for s in corpus["dl-contract-domestic"]}
    assert "rule-19-flight-delays-cancellations" in dl
    intl = {s.section_id for s in corpus["dl-contract-international"]}
    assert "rule-20-flight-delays-cancellations" in intl


def test_table_of_contents_entries_do_not_become_sections():
    corpus = _corpus_sections()
    for doc in ("wn-contract-of-carriage", "dl-contract-domestic", "us-cfr-250-oversales"):
        for s in corpus[doc]:
            assert not re.search(r"\.{4,}", s.heading), f"{doc}: contents line {s.heading!r}"
            assert not re.search(r"\.{4,}", s.text.split("\n", 1)[0]), f"{doc}#{s.section_id}"


def test_prose_starting_with_article_is_not_a_heading():
    """'Article 29 of the Montreal Convention.' is a sentence in UK261 Article 3."""
    ids = {s.section_id for s in _corpus_sections()["uk-261"]}
    assert not any(i.startswith("article-29") for i in ids)
    eu = {s.section_id for s in _corpus_sections()["eu-261-2004"]}
    assert {"article-2-definitions", "article-15-exclusion-of-waiver"} <= eu


def test_list_markers_and_paragraph_numbers_survive_furniture_removal():
    """Repetition-based furniture detection deleted '(a)' from 14 CFR Part 250,
    '1.'/'2.' from EU261, and UK261's paragraph numbers."""
    from flight_delay.ingest import _strip_page_furniture

    corpus = _corpus_sections()
    art3 = next(s for s in corpus["uk-261"] if s.section_id == "article-3-scope")
    assert art3.text.startswith("1\n")
    eu7 = next(s for s in corpus["eu-261-2004"] if s.section_id.startswith("article-7"))
    assert eu7.text.startswith("1.\n")
    pages = [f"Header\nbody {i}\n(a)\nclause {i}\nmore\ntext\nlines\nhere\nmore\nbody\nFooter\n{i + 1}"
             for i in range(5)]
    kept = _strip_page_furniture(pages)
    assert all("(a)" in p and "Header" not in p and "Footer" not in p for p in kept)
    assert not any(p.rstrip().endswith(str(i + 1)) for i, p in enumerate(kept))


def test_cfr_prints_keep_clauses_under_their_own_section():
    """Extraction order filed § 250.8(a) before its heading; the manifest sorts eCFR prints."""
    p250 = {s.section_id: s for s in _corpus_sections()["us-cfr-250-oversales"]}

    def section(prefix):
        return next(s for sid, s in p250.items() if sid.startswith(prefix))

    assert section("250-8-").text.lstrip().startswith("(a)")
    assert "substitution of equipment of lesser capacity" in _flat(section("250-6-").text)
    assert section("250-5-").heading.endswith("involuntarily.")      # wrapped heading completed


def test_embed_header_names_the_document_once():
    sec = next(s for s in _corpus_sections()["eu-261-2004"] if s.section_id.startswith("article-7"))
    ch = chunk_section(sec, target_tokens=256)[0]
    header = ch.embed_text.split("\n\n", 1)[0]
    assert header.count(sec.doc_title) == 1 and "Article 7" in header


def _long_section(n: int = 60):
    from dataclasses import replace

    body = "\n\n".join(
        f"({chr(97 + i % 26)}) The carrier shall provide item {i} to every passenger whose flight "
        f"is delayed; the passenger may also request a refund of the unused fare {i}."
        for i in range(n))
    return replace(_sections()[1], text=body)


def test_chunk_bodies_never_exceed_the_target_in_the_counter_used():
    sec = _long_section()
    for target in (32, 64, 128):
        for counter in (None, lambda t: len(t.split())):
            count = counter or estimate_tokens
            for ch in chunk_section(sec, target_tokens=target, overlap_pct=15, count_tokens=counter):
                assert count(ch.text) <= target
                assert ch.token_estimate == count(ch.text)


def test_overlap_is_carried_but_never_duplicates_a_whole_chunk():
    sec = _long_section()
    chunks = chunk_section(sec, target_tokens=128, overlap_pct=30)   # budget 38 > one ~35-token unit
    assert len(chunks) > 2
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts))
    shared = sum(bool(set(a.split("\n\n")) & set(b.split("\n\n")))
                 for a, b in zip(texts, texts[1:], strict=False))
    assert shared >= 1


def test_an_unsplittable_sentence_is_the_only_oversize_chunk():
    from dataclasses import replace

    long_sentence = " ".join(["word"] * 400)
    sec = replace(_sections()[1], text=f"Short clause one.\n\n{long_sentence}\n\nShort clause two.")
    chunks = chunk_section(sec, target_tokens=32, overlap_pct=0)
    over = [c for c in chunks if c.token_estimate > 32]
    assert len(over) == 1 and over[0].text == long_sentence


def test_ndcg_cutoff_applies_to_chunk_positions():
    """Audit reproduction: five copies of a wrong section, then the gold chunk at
    position 6, scored 0.63 when duplicates were removed before the cutoff."""
    import retrieval_eval as rev

    assert rev.ndcg_at_k(["x", "x", "x", "x", "x", "gold"], {"gold"}, 5) == 0.0
    assert rev.ndcg_at_k(["gold", "gold", "gold"], {"gold"}, 5) == 1.0
    assert rev.ndcg_at_k(["x", "gold", "gold"], {"gold", "other"}, 5) < 1.0


def test_gold_passages_do_not_depend_on_chunk_size():
    """Gold evidence is resolved against sections: no chunk ids, the same text at any size."""
    import build_golden_set as bgs
    from golden import refs

    from flight_delay.ingest import clause_units

    for r in _golden_rows():
        for c in r["context"]:
            assert "chunk_id" not in c
    passages = bgs.resolve(refs.P260_REFUND, _corpus_sections(), lambda s: clause_units(s.text))
    assert len(passages) == 1
    p = passages[0]
    text = p.text(clause_units(p.section.text))
    assert "holds a nonrefundable ticket on a scheduled flight" in _flat(text)
    assert len(text.split()) <= bgs.GOLD_PASSAGE_WORDS or p.first == p.last


def test_overlapping_gold_passages_merge():
    import build_golden_set as bgs

    sec = object()
    merged = bgs.merge_passages([bgs.Passage(sec, 2, 4), bgs.Passage(sec, 5, 6),
                                 bgs.Passage(sec, 9, 9), bgs.Passage(sec, 0, 1)])
    assert [(p.first, p.last) for p in merged] == [(0, 6), (9, 9)]


def test_build_context_budgets_in_request_units_not_embedding_tokens():
    """token_estimate counts embedding-model tokens; the request budget is chars/4."""
    ch = Chunk(chunk_id="c", doc_id="d", doc_title="Doc", publisher="p", jurisdiction="US",
               section_id="s", breadcrumb="Doc > S", text="x" * 400, embed_text="x" * 400,
               source_url="", token_estimate=10_000)
    ctx = build_context([ch], None, token_budget=200, reserve_for_answer=0)
    assert ctx.used_chunks == [ch]
    assert "Doc — S |" in ctx.text                       # the title is not repeated


def test_the_governing_regimes_law_survives_a_tight_context_budget():
    """Session 19, measured on live-01 (UA9901, a Heathrow departure, so UK261
    governs): the balancer reserved UK261 a seat, then build_context dropped it for
    want of ~27 tokens and the answer was built from US law and United's contract
    alone. A guaranteed chunk is in the context because of the ROUTE, so it always
    sorts last on relevance - which made it the FIRST thing the budget discarded,
    exactly inverting the guarantee. The passenger was owed GBP 520 under UK261."""
    def chunk(name, size, *, guaranteed=False, jurisdiction="US"):
        return Chunk(chunk_id=name, doc_id=name, doc_title=name, publisher="p",
                     jurisdiction=jurisdiction, section_id=name, breadcrumb="b",
                     text="x" * size, embed_text="x" * size, source_url="",
                     guaranteed=guaranteed)

    # best-first, as the balancer hands them over; the governing law ranks last
    strong = [chunk(f"us-{i}", 4000) for i in range(4)]
    uk = chunk("uk-261", 2000, guaranteed=True, jurisdiction="UK")
    ctx = build_context([*strong, uk], None, token_budget=5000, reserve_for_answer=700)

    kept = {c.chunk_id for c in ctx.used_chunks}
    assert "uk-261" in kept, "the governing regime's law must survive the budget"
    assert len(kept) < 5, "it should displace a weaker source, not be added for free"
    # relevance order is still what decides POSITION: sources are emitted
    # weakest-first so the best sits next to the question, and admitting the
    # guaranteed chunk early must not promote it past better-ranked text.
    order = [c.chunk_id for c in ctx.used_chunks]
    assert order == sorted(order, key=lambda n: -[*[c.chunk_id for c in [*strong, uk]]].index(n))

    # and it is still best-effort: one that cannot fit on its own is skipped rather
    # than emptying the context
    huge = chunk("uk-261-huge", 40_000, guaranteed=True, jurisdiction="UK")
    ctx2 = build_context([*strong, huge], None, token_budget=5000, reserve_for_answer=700)
    assert ctx2.used_chunks and "uk-261-huge" not in {c.chunk_id for c in ctx2.used_chunks}


def test_balancer_marks_the_governing_regimes_chunk_as_guaranteed():
    """The flag is what build_context keys on, so it must be set where the seat is
    reserved - not inferred later from the jurisdiction, which cannot tell a chunk
    that was reserved from one that simply out-ranked everything."""
    from flight_delay.retrieval import balance_by_source_class

    def chunk(name, cls, juris):
        return Chunk(chunk_id=name, doc_id=name, doc_title=name, publisher="p",
                     jurisdiction=juris, section_id=name, breadcrumb="b", text="x" * 100,
                     embed_text="x", source_url="", airline_iata="UA" if cls == "airline" else "")

    ranked = [chunk("ua-1", "airline", "US"), chunk("ua-2", "airline", "US"),
              chunk("us-law", "government", "US"), chunk("us-law-2", "government", "US"),
              chunk("uk-law", "government", "UK")]
    # production is final_k=8 with 2+2 quotas, so there is room past the quotas
    out = balance_by_source_class(ranked, 5, min_government=2, min_airline=2,
                                  require_jurisdictions=("US", "UK"))
    picked = {c.chunk_id for c in out}
    assert "uk-law" in picked
    assert next(c for c in out if c.chunk_id == "uk-law").guaranteed is True
    # US was already satisfied by the government quota, so it is not re-reserved
    assert next(c for c in out if c.chunk_id == "us-law").guaranteed is False

    # KNOWN EDGE, asserted so a change is deliberate: when min_government +
    # min_airline already fill top_k there is no slot left and the guarantee cannot
    # fire. It does not bite in production (final_k 8, quotas 2 + 2), but shrinking
    # final_k to 4 would silently drop the governing regime's law again.
    tight = balance_by_source_class(ranked, 4, min_government=2, min_airline=2,
                                    require_jurisdictions=("US", "UK"))
    assert "uk-law" not in {c.chunk_id for c in tight}


def test_stage_one_scores_the_context_production_kept():
    """Evidence metrics come from the answer's BuiltContext, ranking metrics from
    the reranked pool; no model is called and nothing touches a database."""
    import retrieval_eval as rev

    from flight_delay.retrieval import NoopReranker

    base = Settings(_env_file=None, embedder="hash", reranker="none", embedding_dim=256)
    items = [r for r in _golden_rows() if r["id"] in ("dom-01", "eu-us-01")]
    summary, rows = rev.evaluate_config(base, 256, 15, items, HashEmbedder(dim=256), NoopReranker(),
                                        backend="memory")
    assert [r["id"] for r in rows] == ["dom-01", "eu-us-01"]
    for r in rows:
        assert r["context_built"] and r["turn_outcome"] == "llm_error"   # the stage-1 stop
        assert 0 < r["n_kept"] <= r["n_retrieved"] <= r["n_ranked"]
        assert r["context_tokens"] > 0
    assert summary["embed_truncated_pct"] is None           # no model tokenizer on a smoke run
    assert "dense_search" not in summary                    # Postgres-only diagnostic


def test_user_conversation_confirms_the_flight_and_keeps_it_for_follow_ups():
    """Session 17 transcript: ask -> clarification -> flight number -> "do you know my
    airports?". The lookup must be confirmed in words, and the follow-up must still see
    the flight data (it used to lose it and fail validation)."""
    from flight_delay.pipeline import RagPipeline

    class Memory(_FakeStore):
        states: dict = {}

        def load_state(self, conv_id):
            return self.states.get(conv_id)

        def save_state(self, conv_id, state):
            self.states[conv_id] = state

    class Recorder:
        def __init__(self):
            self.prompts = []

        def complete(self, system, user):
            self.prompts.append(user)
            return "DL216 is from JFK to DSS."

    dl216 = FlightStatus(
        flight_iata="DL216", airline_iata="DL", status="cancelled", dep_iata="JFK",
        dep_time=None, dep_estimated=None, dep_delayed=None, arr_iata="DSS",
        arr_time=None, arr_estimated=None, arr_delayed=None, delayed=None)
    llm = Recorder()
    settings = Settings(_env_file=None, embedder="hash", reranker="none",
                        confidence_gate_enabled=False)
    pipe = RagPipeline(_FakeRetriever(), llm, _FakeTool({"DL216": dl216}), settings, store=Memory())

    first = pipe.run("my delta flight is cancelled . what to do?", conversation_id="c1")
    assert first.clarification
    second = pipe.run("flight no DL216. Flying to Senegal.", conversation_id="c1")
    assert second.text.startswith("I found DL216: JFK to DSS, cancelled. If that's not your flight")
    third = pipe.run("do you know my flight departure and arrival airport?", conversation_id="c1")
    assert "JFK" in llm.prompts[-1] and "DSS" in llm.prompts[-1]      # flight data still sent
    assert not third.text.startswith("I found")                        # confirmed once, not every turn


def _wn1922(status="scheduled", delay=None):
    return FlightStatus(
        flight_iata="WN1922", airline_iata="WN", status=status, dep_iata="SJD",
        dep_time="2026-09-17 13:05", dep_estimated=None, dep_delayed=delay, arr_iata="PHX",
        arr_time="2026-09-17 14:10", arr_estimated=None, arr_delayed=delay, delayed=delay,
    )


def test_flight_data_without_a_delay_contradicts_a_passenger_who_says_delayed():
    # Session 18 transcript: "is my southwest flight delayed?" then "WN1922"; AirLabs had it
    # scheduled with no delay, and the answer explained delay rights as if it were delayed.
    from datetime import date

    from flight_delay.pipeline import plan_turn

    history = [("user", "is my southwest flight delayed?"), ("assistant", "I couldn't put together an answer")]
    plan = plan_turn("so the southwest flight no is WN1922", history, lambda n: _wn1922(),
                     today=date(2026, 9, 17))
    assert plan.status_conflict is True
    assert plan.flight_note.startswith(
        "I found WN1922: SJD to PHX, scheduled, no delay reported. If that's not your flight, tell me. "
        "From what I can find, WN1922 isn't delayed: the flight data shows it scheduled, no delay "
        "reported (departure 2026-09-17 13:05 local time from SJD).")
    assert "if Southwest Airlines has told you it is delayed, tell me what they said" in plan.flight_note


def test_no_conflict_when_the_data_shows_the_delay_or_the_passenger_mentions_none():
    from datetime import date

    from flight_delay.pipeline import plan_turn

    today = date(2026, 9, 17)
    delayed = plan_turn("My flight WN1922 is delayed, what am I owed?", [], lambda n: _wn1922(delay=95),
                        today=today)
    assert delayed.status_conflict is False and "delayed about 1 h 35 min" in delayed.flight_note
    plain = plan_turn("What gate does WN1922 leave from?", [], lambda n: _wn1922(), today=today)
    assert plain.status_conflict is False
    cancelled = plan_turn("My flight WN1922 was cancelled", [], lambda n: _wn1922("cancelled"), today=today)
    assert cancelled.status_conflict is False
    # A record for another day says nothing about the passenger's flight's status.
    other_day = plan_turn("My flight WN1922 yesterday was delayed", [], lambda n: _wn1922(), today=today)
    assert other_day.status_conflict is False and "isn't delayed" not in other_day.flight_note


def test_a_cancellation_claim_is_contradicted_by_a_delayed_record():
    from datetime import date

    from flight_delay.pipeline import plan_turn

    plan = plan_turn("my flight WN1922 got cancelled", [], lambda n: _wn1922("active", 40),
                     today=date(2026, 9, 17))
    assert plan.status_conflict is True
    assert "WN1922 isn't cancelled: the flight data shows it active, delayed about 40 min." in plan.flight_note


def test_pipeline_tells_the_passenger_and_the_model_the_flight_is_not_delayed():
    from flight_delay.pipeline import RagPipeline

    llm = _ScriptedLLM("A refund is owed when the airline cancels [S1].")
    s = Settings(_env_file=None, embedder="hash", reranker="none", confidence_gate_enabled=False)
    pipe = RagPipeline(_OneChunkRetriever(), llm, _FakeTool({"WN1922": _wn1922()}), s, store=_FakeStore())
    a = pipe.run("Is my Southwest flight WN1922 delayed?")
    assert "From what I can find, WN1922 isn't delayed" in a.text
    assert "do not describe their flight as delayed or cancelled" in llm.prompts[0]


def test_turn_log_keeps_the_rejected_draft_and_the_providers_limit(caplog):
    """Session 19: a rejected answer used to exist nowhere. It is not sent to the
    passenger (right) and not stored in `messages` (right - that table is the
    conversation), so the only trace was the uncited sentence list. Every rejection
    then cost a full generation and taught nothing. Same for a 429: status plus
    Retry-After cannot tell tokens-per-minute from tokens-per-day."""
    import json as _json
    import logging

    from flight_delay.models import Answer
    from flight_delay.pipeline import _log_turn

    answer = Answer(
        question="My United flight from London Heathrow to Chicago was delayed 5 hours.",
        text="safe fallback shown to the passenger",
        outcome="validation_failed",
        intent="policy",
        retry_count=1,
        validation_failures=["2 factual sentence(s) carry no citation"],
        timings_ms={"total": 45482.0},
        diagnostics={
            "uncited_factual_sentences": ["Ask United for the GBP 520 compensation claim form."],
            "rejected_answer": "Under UK261 you are owed GBP 520 [S1]. Ask United for the form.",
            "usage": [{"prompt_tokens": 6444, "error_status": 429, "error_code": "rate_limit_exceeded",
                       "retry_after_s": 2915.0,
                       "error": "LLMTransientError: ... on tokens per day (TPD): Limit 200000"}],
        },
    )
    with caplog.at_level(logging.INFO, logger="flight_delay.turn"):
        _log_turn(answer)
    logged = _json.loads(caplog.records[-1].getMessage().split("turn ", 1)[1])

    assert logged["rejected_answer"].startswith("Under UK261")
    assert logged["uncited"] == ["Ask United for the GBP 520 compensation claim form."]
    call = logged["calls"][0]
    assert call["error_status"] == 429 and call["retry_after_s"] == 2915.0
    assert "tokens per day (TPD)" in call["error"], "a 429 must name its own limit"


# ---------------------------------------------------------------- monitoring
# Session 20 (Step 7): the dead rag_eval_recall_at_20 alert was a rule on a
# metric nothing emitted - it could never fire, and it made the alert list look
# like coverage that did not exist. These two tests make that class of mistake
# fail in CI instead of at 3am, and they are cheap: both are pure text/YAML.

def _emitted_metric_names() -> set[str]:
    """Metric names src/flight_delay/metrics.py actually registers, with the
    suffixes prometheus_client derives from them (_bucket/_sum/_count)."""
    src = (ROOT / "src" / "flight_delay" / "metrics.py").read_text(encoding="utf-8")
    base = set(re.findall(r'\n\w+ = (?:Counter|Gauge|Histogram)\(\s*\n?\s*"([a-z_:]+)"', src))
    assert len(base) > 15, "the metric definitions stopped matching this pattern"
    names: set[str] = set()
    for n in base:
        names |= {n, f"{n}_bucket", f"{n}_sum", f"{n}_count", f"{n}_created", f"{n}_total"}
        if n.endswith("_total"):        # Counter("x_total") also exposes x_created
            names |= {n[:-6], f"{n[:-6]}_created"}
    return names


def test_dashboard_and_alerts_only_use_metrics_the_app_emits():
    """Every series named in the DEFAULT dashboard and alert rules must exist in
    metrics.py. Series belonging to the optional self-hosted profile (vllm:*)
    must NOT appear there - a hosted provider reports no KV cache, so a panel or
    alert reading those is permanently blank, which reads as a broken system
    rather than as an absent one."""
    import json as _json

    emitted = _emitted_metric_names()
    # PromQL keywords, aggregation labels and label VALUES are not metric names.
    non_metrics = {
        "histogram_quantile", "clamp_min", "rate", "irate", "increase", "absent",
        "sum", "avg", "count", "topk", "vector", "scalar", "time", "group_left",
        "group_right", "ignoring", "without", "unless", "bool", "offset",
        "intent", "outcome", "stage", "reason", "source", "kind", "severity",
        "validation_failed", "unknown_marker", "request_error", "transient_error",
        "malformed", "empty", "truncated", "instance",
    }
    dashboard = _json.loads(
        (ROOT / "deploy" / "grafana" / "dashboards" / "rag.json").read_text(encoding="utf-8"))
    exprs = [t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])]
    rules_text = (ROOT / "deploy" / "prometheus" / "rules" / "rag.yml").read_text(encoding="utf-8")
    exprs += re.findall(r"expr:\s*(.+)", rules_text)

    unknown = set()
    for expr in exprs:
        for token in re.findall(r"\b([a-z][a-z0-9_]{4,})\b", expr):
            if token not in non_metrics and token not in emitted:
                unknown.add(token)
    assert not unknown, f"dashboard/alerts reference series the app never emits: {sorted(unknown)}"
    assert "vllm:" not in "".join(exprs), "vllm series belong in the -optional files"


def test_prometheusrule_manifest_is_generated_from_the_local_rules():
    """The cluster reads alerts from a PrometheusRule CR, not from the file
    Prometheus mounts locally, so the thresholds exist twice. One copy is
    generated from the other; this fails if someone edits either by hand and the
    two drift apart - the copy nobody reads being the one that pages."""
    import make_prometheusrule

    target = ROOT / "deploy" / "k8s" / "45-alert-rules.yaml"
    assert target.exists(), "run scripts/make_prometheusrule.py"
    assert target.read_text(encoding="utf-8") == make_prometheusrule.render(), (
        "deploy/k8s/45-alert-rules.yaml is stale: re-run scripts/make_prometheusrule.py")


def test_k8s_configmap_keys_are_settings_the_app_reads():
    """A ConfigMap key the app does not read is not an error anywhere: pydantic
    ignores it, the pod starts, and the cluster quietly runs a configuration
    nobody chose. This catches the typo that `kubectl apply` cannot - and it is
    the offline half of validating these manifests, since schema validation
    needs a live API server."""
    import yaml

    from flight_delay.config import Settings

    docs = list(yaml.safe_load_all(
        (ROOT / "deploy" / "k8s" / "18-config.yaml").read_text(encoding="utf-8")))
    cfg = next(d for d in docs if d["kind"] == "ConfigMap")

    known = set(Settings.model_fields)
    for field in Settings.model_fields.values():
        alias = getattr(field, "validation_alias", None)
        for choice in getattr(alias, "choices", []) or ([alias] if isinstance(alias, str) else []):
            known.add(str(choice))

    unknown = {k for k in cfg["data"] if k.lower() not in known}
    assert not unknown, f"ConfigMap sets values the app never reads: {sorted(unknown)}"

    # The settings that decide which evidence is retrieved must equal the ones
    # this repository was measured with. A cluster running other chunk settings
    # does not degrade quietly: /ready refuses the index and serves 503.
    assert cfg["data"]["CHUNK_TARGET_TOKENS"] == "512"
    assert cfg["data"]["CHUNK_OVERLAP_PCT"] == "15"
    # The gate is uncalibrated; it is off in .env and must be off here too.
    assert cfg["data"]["CONFIDENCE_GATE_ENABLED"] == "false"
    # Recorded flight responses must never be served to a passenger.
    assert cfg["data"]["USE_FLIGHT_FIXTURES"] == "false"
    # A reasoning extra_body without an explicit cap is refused by config.py.
    assert "reasoning" in cfg["data"]["LLM_EXTRA_BODY"]
    assert cfg["data"]["LLM_MAX_COMPLETION_TOKENS"] == "900"


def test_k8s_manifests_pin_images_and_name_the_same_secret_keys():
    """Two failure modes that only show up in the cluster: a floating :latest tag
    (two pods of one Deployment running different builds, and nothing to roll
    back to), and a secretKeyRef naming a key the documented Secret does not
    create (pod stuck in CreateContainerConfigError)."""
    import yaml

    documented = {"postgres-password", "pg-dsn", "llm-api-key", "airlabs-key", "conversation-secret"}
    runbook = (ROOT / "RUNBOOK_AWS.md").read_text(encoding="utf-8")
    for key in documented:
        assert f"--from-literal={key}=" in runbook, f"RUNBOOK_AWS.md no longer creates {key}"
    referenced: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            if "secretKeyRef" in node:
                referenced.add(node["secretKeyRef"]["key"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for path in sorted((ROOT / "deploy" / "k8s").glob("*.yaml")):
        if path.name == "eksctl-cluster.yaml":
            continue
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if not doc:
                continue
            walk(doc)
            spec = doc.get("spec", {}) or {}
            if doc.get("kind") == "CronJob":
                spec = spec["jobTemplate"]["spec"]
            pod = (spec.get("template", {}) or {}).get("spec", {}) or {}
            for container in (pod.get("containers") or []) + (pod.get("initContainers") or []):
                image = container["image"]
                assert ":" in image.rsplit("/", 1)[-1] or "@sha256:" in image, f"{path.name}: {image} has no tag"
                assert not image.endswith(":latest"), f"{path.name}: {image} is unpinned"

    assert referenced <= documented, (
        f"manifests read Secret keys RUNBOOK_AWS.md never creates: {sorted(referenced - documented)}")
    assert "conversation-secret" in referenced, "20-api.yaml no longer reads CONVERSATION_SECRET"


def test_k8s_preflight_static_checks_pass_on_the_repository():
    """The offline half of what `make eks-deploy` checks: a shared
    CONVERSATION_SECRET for a multi-replica API, vLLM's window >= the API's
    context + completion cap, one image for the API and the index Job, every
    claim on a StorageClass the repository defines, the Service handed to the
    AWS Load Balancer Controller with client IPs preserved, and every pod that
    reads PG_DSN admitted by the Postgres NetworkPolicy. Each of these parsed
    and applied cleanly while broken (PROGRESS.md Session 33)."""
    import k8s_preflight

    assert k8s_preflight.static_problems() == []


def test_k8s_preflight_catches_the_defects_it_names(monkeypatch):
    """Break each property in memory and check the preflight says so - a check
    that never fails is not a check."""
    import copy

    import k8s_preflight

    original = k8s_preflight._docs

    def broken(name):
        docs = copy.deepcopy(original(name))
        for d in docs:
            if name == "20-api.yaml" and d["kind"] == "Deployment":
                c = d["spec"]["template"]["spec"]["containers"][0]
                c["env"] = [e for e in c["env"] if e["name"] != "CONVERSATION_SECRET"]
                d["spec"]["template"]["spec"]["serviceAccountName"] = "fdr-backup"   # S3 role on the API
            if name == "50-index-job.yaml":
                del d["spec"]["template"]["spec"]["automountServiceAccountToken"]
            if name == "19-api-service.yaml" and d["kind"] == "Service":
                d["metadata"]["annotations"] = {}
            if name == "30-vllm.yaml" and d["kind"] == "Deployment":
                args = d["spec"]["template"]["spec"]["containers"][0]["args"]
                args[args.index("--max-model-len") + 1] = "10880"   # the old, 9728-based value
            if name == "10-postgres.yaml" and d["kind"] == "StatefulSet":
                d["spec"]["volumeClaimTemplates"][0]["spec"]["storageClassName"] = "gp2"
            if name == "60-db-backup.yaml" and d["kind"] == "CronJob":
                d["spec"]["jobTemplate"]["spec"]["template"]["metadata"]["labels"]["app"] = "other"
        return docs

    monkeypatch.setattr(k8s_preflight, "_docs", broken)
    text = "\n".join(k8s_preflight.static_problems())
    assert "CONVERSATION_SECRET" in text
    assert "--max-model-len 10880" in text
    assert "'gp2'" in text
    assert "Classic Load Balancer" in text
    assert "db-backup reads PG_DSN" in text
    assert "api runs as IRSA account 'fdr-backup'" in text
    assert "index-corpus mounts a ServiceAccount token" in text

    # A LoadBalancer Service back in the Deployment's file loses the ordering
    # that gives the pods the NLB readiness gate.
    def merged(name):
        docs = original(name)
        return docs + original("19-api-service.yaml") if name == "20-api.yaml" else docs

    monkeypatch.setattr(k8s_preflight, "_docs", merged)
    assert any("outside 19-api-service.yaml" in p for p in k8s_preflight.check_load_balancer())


def test_k8s_preflight_placeholder_scan_finds_the_shipped_markers():
    """The repository ships ACCOUNT_ID / REPLACE_WITH_* on purpose; the deploy
    machine must replace them. The scan must find them in live YAML and ignore
    them in comments (the commented TLS annotation carries one)."""
    import k8s_preflight

    deploy = "\n".join(k8s_preflight.placeholder_problems("deploy"))
    for marker in ("ACCOUNT_ID", "REPLACE_WITH_IMMUTABLE_TAG", "REPLACE_WITH_S3_BUCKET"):
        assert marker in deploy
    assert "REPLACE_WITH_CERT_ID" not in deploy
    # The SNS topic ARN in the monitoring values carries the account id.
    assert "ACCOUNT_ID" in "\n".join(k8s_preflight.placeholder_problems("monitoring"))


def test_eksctl_cluster_declares_what_the_manifests_rely_on():
    """The manifests assume three things the CLUSTER must provide: the EBS CSI
    driver (05-storageclass.yaml's provisioner), network policy enforcement
    (15-networkpolicy.yaml is otherwise accepted and ignored), and an IRSA
    ServiceAccount for the AWS Load Balancer Controller (without it the API
    Service becomes a Classic ELB or stays pending)."""
    import json

    import yaml

    cfg = yaml.safe_load((ROOT / "deploy" / "k8s" / "eksctl-cluster.yaml").read_text(encoding="utf-8"))
    addons = {a["name"]: a for a in cfg["addons"]}
    assert "aws-ebs-csi-driver" in addons
    assert json.loads(addons["vpc-cni"]["configurationValues"])["enableNetworkPolicy"] == "true"
    accounts = {(s["metadata"]["namespace"], s["metadata"]["name"]): s for s in cfg["iam"]["serviceAccounts"]}
    lbc = accounts[("kube-system", "aws-load-balancer-controller")]
    assert lbc["wellKnownPolicies"]["awsLoadBalancerController"] is True
    # Least privilege: in the app namespace only the backup job has an AWS role,
    # and Terraform gives that role PutObject under db-backups/ and nothing else.
    assert {n for ns, n in accounts if ns == "fdr"} == {"fdr-backup"}
    assert accounts[("fdr", "fdr-backup")]["attachPolicyARNs"] == [
        "arn:aws:iam::ACCOUNT_ID:policy/fdr-db-backup"]
    tf = (ROOT / "deploy" / "terraform" / "main.tf").read_text(encoding="utf-8")
    # Every AWS action any workload role may take, in the whole file: the
    # backup's one S3 write and Alertmanager's one SNS publish.
    actions = set(re.findall(r'"((?:s3|sns|sts|iam|kms):[A-Za-z*]+)"', tf))
    assert actions == {"s3:PutObject", "sns:Publish"}, actions
    backup = tf.split('resource "aws_iam_policy" "db_backup"')[1].split("\nresource ")[0]
    assert '"s3:PutObject"' in backup and '${aws_s3_bucket.artifacts.arn}/db-backups/*' in backup
    alerts = tf.split('resource "aws_iam_policy" "alerts_publish"')[1].split("\nresource ")[0]
    assert '"sns:Publish"' in alerts and "aws_sns_topic.alerts.arn" in alerts
    # Outside the app namespace, exactly one more workload identity: Alertmanager's.
    assert {(ns, n) for ns, n in accounts if ns not in {"fdr", "kube-system"}} == {("monitoring", "fdr-alertmanager")}
    assert accounts[("monitoring", "fdr-alertmanager")]["attachPolicyARNs"] == [
        "arn:aws:iam::ACCOUNT_ID:policy/fdr-alerts-publish"]
    # A pinned version in extended support bills the control plane at 6x;
    # 1.31-1.33 were already there on 2026-09-20 (describe-cluster-versions).
    assert cfg["metadata"]["version"] not in {"1.31", "1.32", "1.33"}


def test_fdr_config_lives_once_in_18_config_and_is_applied_before_the_index_job():
    """The index Job reads fdr-config with envFrom and runs to completion before
    the API exists. While fdr-config lived in 20-api.yaml - applied AFTER the
    Job - a new cluster's index pod could never start
    (CreateContainerConfigError: configmap "fdr-config" not found)."""
    import yaml

    k8s = ROOT / "deploy" / "k8s"
    homes = []
    for path in sorted(k8s.glob("*.yaml")):
        if path.name == "eksctl-cluster.yaml":
            continue
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if doc and doc.get("kind") == "ConfigMap" and doc["metadata"]["name"] == "fdr-config":
                homes.append(path.name)
    assert homes == ["18-config.yaml"], homes
    assert "kind: ConfigMap" not in (k8s / "20-api.yaml").read_text(encoding="utf-8")

    job = next(d for d in yaml.safe_load_all((k8s / "50-index-job.yaml").read_text(encoding="utf-8")) if d)
    env_from = job["spec"]["template"]["spec"]["containers"][0]["envFrom"]
    assert {"configMapRef": {"name": "fdr-config"}} in env_from

    import k8s_preflight

    order = k8s_preflight._eks_deploy_applies()
    assert order.index("18-config.yaml") < order.index("50-index-job.yaml") < order.index("20-api.yaml")
    assert order.index("19-api-service.yaml") < order.index("<wait-tgb>") < order.index("20-api.yaml")


def _pod_env(manifest: str, secret_value: str = "from-secret") -> dict[str, str]:
    """The environment a Deployment/Job container starts with: fdr-config, then its env."""
    import yaml

    k8s = ROOT / "deploy" / "k8s"
    cm = next(d for d in yaml.safe_load_all((k8s / "18-config.yaml").read_text(encoding="utf-8"))
              if d and d.get("kind") == "ConfigMap")
    doc = next(d for d in yaml.safe_load_all((k8s / manifest).read_text(encoding="utf-8"))
               if d and d.get("kind") in ("Job", "Deployment"))
    c = doc["spec"]["template"]["spec"]["containers"][0]
    env = {k: str(v) for k, v in cm["data"].items()}
    for e in c.get("env", []):
        env[e["name"]] = str(e["value"]) if "value" in e else secret_value
    return env


def test_the_index_job_environment_loads_settings(monkeypatch):
    """Session 37, first real deploy: the index pod died in ~2 s with 'llm_provider
    is groq but LLM_API_KEY is not set' - it loads the API's Settings but was given
    only PG_DSN. Offline checks never built its environment; this does."""
    from flight_delay.config import Settings

    for name in list(Settings.model_fields):
        monkeypatch.delenv(name.upper(), raising=False)
    env = _pod_env("50-index-job.yaml")
    assert env["LLM_PROVIDER"] == "groq"
    # A placeholder, never the real key: the indexer makes no LLM call.
    assert env["LLM_API_KEY"] == "unused-by-indexer"
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=None)
    assert s.pg_dsn == "from-secret" and s.test_double_problems() == []


def test_preflight_catches_the_configmap_applied_after_the_index_job(monkeypatch, tmp_path):
    """Replay the original layout - fdr-config back inside 20-api.yaml and the
    Makefile without 18-config.yaml - and require the preflight to name it."""
    import k8s_preflight

    original = k8s_preflight._docs

    def old_layout(name):
        if name == "18-config.yaml":
            return []
        if name == "20-api.yaml":
            return original("18-config.yaml") + original("20-api.yaml")
        return original(name)

    makefile = tmp_path / "Makefile"
    makefile.write_text(k8s_preflight.MAKEFILE.read_text(encoding="utf-8")
                        .replace("\tkubectl apply -f deploy/k8s/18-config.yaml\n", ""), encoding="utf-8")
    monkeypatch.setattr(k8s_preflight, "_docs", old_layout)
    monkeypatch.setattr(k8s_preflight, "MAKEFILE", makefile)
    text = "\n".join(k8s_preflight.check_config_order())
    assert "fdr-config must be defined exactly once, in 18-config.yaml" in text
    assert "applies 50-index-job.yaml BEFORE 20-api.yaml" in text
    assert "CreateContainerConfigError" in text


def test_preflight_env_parity_is_explicit_about_a_missing_env(tmp_path, capsys):
    """A sanitized copy has no .env: the parity check is skipped WITH a warning
    (never a silent OK), and --require-env - what `make preflight` passes on
    the deploying machine - turns the absence into a failure."""
    import k8s_preflight

    missing = tmp_path / ".env"
    assert k8s_preflight.main(["--static-only", "--env", str(missing)]) == 0
    assert "parity check skipped" in capsys.readouterr().out
    assert k8s_preflight.main(["--static-only", "--env", str(missing), "--require-env"]) == 1
    assert "--require-env" in (ROOT / "Makefile").read_text(encoding="utf-8")


def test_runbook_points_tls_and_config_edits_at_the_right_files():
    """After the Service and ConfigMap splits, the runbook must send TLS edits
    to 19-api-service.yaml and LLM_* edits to 18-config.yaml, and create the
    namespace idempotently (eksctl may already have created it)."""
    runbook = (ROOT / "RUNBOOK_AWS.md").read_text(encoding="utf-8")
    https = runbook.split("## 5b.")[1].split("\n## ")[0]
    assert "19-api-service.yaml" in https and "20-api.yaml" not in https
    assert "kubectl apply -f deploy/k8s/19-api-service.yaml" in https
    assert "switching LLM_* in deploy/k8s/18-config.yaml" in runbook
    assert "kubectl create namespace fdr\n" not in runbook
    assert "kubectl create namespace fdr --dry-run=client -o yaml | kubectl apply -f -" in runbook


def test_alerts_reach_email_through_sns_with_no_mail_password(monkeypatch):
    """Prometheus -> Alertmanager -> SNS -> email. No SMTP receiver and no mail
    password anywhere; Alertmanager uses the eksctl IRSA ServiceAccount (a
    chart-made one has no AWS identity) and publishes to the Terraform topic,
    which has an email subscription to var.alert_email."""
    import copy

    import k8s_preflight

    assert k8s_preflight.check_alert_delivery() == []
    tf = (ROOT / "deploy" / "terraform" / "main.tf").read_text(encoding="utf-8")
    assert 'resource "aws_sns_topic" "alerts"' in tf and 'name = "fdr-alerts"' in tf
    sub = tf.split('resource "aws_sns_topic_subscription" "alerts_email"')[1].split("\n}")[0]
    assert 'protocol  = "email"' in sub and "var.alert_email" in sub
    values_text = (ROOT / "deploy" / "helm" / "kube-prometheus-stack-values.yaml").read_text(encoding="utf-8")
    assert "alertmanager-smtp" not in (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "alertmanager-smtp" not in (ROOT / "RUNBOOK_AWS.md").read_text(encoding="utf-8")

    # Break it two ways; the preflight must notice both.
    import yaml
    real = yaml.safe_load(values_text)
    broken = copy.deepcopy(real)
    broken["alertmanager"]["serviceAccount"]["create"] = True
    broken["alertmanager"]["config"]["global"] = {"smtp_smarthost": "smtp.gmail.com:587"}
    original_load = yaml.safe_load

    def fake_load(stream):
        text = stream if isinstance(stream, str) else stream.read()
        return broken if "alertmanager:" in text and "sns_configs" in text else original_load(text)

    monkeypatch.setattr(k8s_preflight.yaml, "safe_load", fake_load)
    text = "\n".join(k8s_preflight.check_alert_delivery())
    assert "must use the eksctl IRSA ServiceAccount" in text
    assert "SMTP/email receiver is back" in text


def test_makefile_pins_every_helm_chart():
    """`helm upgrade --install` without --version installs whatever is newest
    that day, and never upgrades the chart's CRDs. Every chart the Makefile
    installs must carry a pinned version variable with a concrete value."""
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    installs = re.findall(r"helm upgrade --install (\S+) (\S+) \\\n\t  --version \$\((\w+)\)", text)
    assert len(installs) == text.count("helm upgrade --install") == 2, installs
    for _release, _chart, var in installs:
        assert re.search(rf"^{var} \?= \d+\.\d+\.\d+$", text, re.M), f"{var} has no pinned version"


def test_ready_refreshes_the_indexed_chunks_gauge(monkeypatch):
    """The index is built by a separate process, so a pod that starts before the
    corpus exists reads 0 at startup and would report 0 forever - while serving
    real answers. That combination pages IndexEmpty (critical) on a healthy
    deployment and shows "0 chunks indexed" on the dashboard. Observed in the
    local container stack with 571 chunks actually indexed."""
    from fastapi.testclient import TestClient
    from prometheus_client import REGISTRY

    import flight_delay.api as api

    settings = Settings(_env_file=None, embedder="hash", reranker="none")

    class GrowingStore(MemoryStore):
        n = 0

        def count(self):
            return self.n

        def active_identity(self):
            return _identity_for(settings)

    store = GrowingStore()
    fake = type("P", (), {"retriever": type("R", (), {"store": store})()})()
    monkeypatch.setitem(api.STATE, "pipeline", fake)
    monkeypatch.setitem(api.STATE, "settings", settings)
    client = TestClient(api.app)

    api.index_chunks_total.set(0)          # what startup saw: no corpus yet
    assert client.get("/ready").status_code == 503

    store.n = 571                          # the index Job finished
    assert client.get("/ready").status_code == 200
    assert REGISTRY.get_sample_value("index_chunks_total") == 571.0


def test_startup_publishes_the_airlabs_quota_before_any_lookup():
    """A Gauge that has never been set reports 0, and 0 remaining calls reads as
    'quota exhausted' - AirLabsQuotaLow (warning) fired on a freshly started
    stack that had made no AirLabs call at all. An unmeasured gauge must not be
    readable as an alarming measurement."""
    from prometheus_client import REGISTRY

    from flight_delay.metrics import airlabs_quota_remaining

    airlabs_quota_remaining.set(0)
    assert REGISTRY.get_sample_value("airlabs_quota_remaining") == 0.0

    # What the lifespan does, with a stand-in quota counter.
    quota = type("Q", (), {"remaining": lambda self: 940})()
    airlabs_quota_remaining.set(quota.remaining())
    assert REGISTRY.get_sample_value("airlabs_quota_remaining") == 940.0

    src = (ROOT / "src" / "flight_delay" / "api.py").read_text(encoding="utf-8")
    assert "airlabs_quota_remaining.set(quota.remaining())" in src, (
        "startup must publish the quota; see the alert rule AirLabsQuotaLow")


# ==========================================================================
# Session 22: corpus cleanup, structured retrieval query, topic filter, section cap
# ==========================================================================


def test_legislation_annotations_are_removed_and_paragraph_numbers_kept():
    """UK261 Art 6(3) reached the model as "[F143" (marker F14 + paragraph 3), and
    the Textual Amendments blocks were 17% of UK261's indexed text."""
    from flight_delay.ingest import strip_legislation_annotations

    text = "\n".join([
        "Article 6", "Delay", "b", "for three hours or more in the case of all F13... flights;",
        "[F143", "In case of a delay of three hours or more ... Article 7.]",
        "Textual Amendments", "F13", "Words in Art. 6(1)(b) omitted (31.12.2020) by S.I. 2019/278",
        "F14", "Art. 6(3)(4) inserted (14.12.2023) by S.I. 2023/1370",
        "Article 7", "[F151", "Where reference is made to this Article", "OF12 is not a marker",
        "Textual Amendments", "F15", "Art. 7(1) substituted (31.12.2020)",
        "Article 8", "[F991 an id no block defines stays",
    ])
    out = strip_legislation_annotations(text)
    assert "Textual Amendments" not in out and "S.I." not in out
    assert "\n3\nIn case of a delay" in out            # F14 removed, paragraph 3 kept
    assert "\n1\nWhere reference" in out               # F15 removed, paragraph 1 kept
    assert "all flights;" in out.replace("  ", " ")    # "F13..." (omitted words) removed
    assert "OF12 is not a marker" in out and "[F991" in out  # only defined ids, whole markers
    assert strip_legislation_annotations("F14 plain text") == "F14 plain text"


def test_official_journal_footnotes_are_cut_but_the_footer_stays_for_furniture():
    from flight_delay.ingest import _drop_after_page_number

    page = "\n".join(["Body text (99).", "EN", "OJ C, 25.9.2024", "22/36",
                      "ELI: http://data.europa.eu/eli/C/2024/5687/oj",
                      "(99) Joined Cases C-402/07 and C-432/07, Sturgeon", "continued citation"])
    out = _drop_after_page_number(page)
    assert out.endswith("ELI: http://data.europa.eu/eli/C/2024/5687/oj")
    assert "Sturgeon" not in out
    assert _drop_after_page_number("no page number\n(1) a list item") == "no page number\n(1) a list item"


def test_decimal_outline_headings_become_sections_and_contents_entries_do_not():
    from flight_delay.ingest import _split_into_sections

    body = "Passengers reaching their final destination with a delay of 3 hours or more are compensated."
    text = "\n".join([
        "TABLE OF CONTENTS",
        "4.4.7.", "'Long delays' at arrival . . . . . . . . . . 22",
        "4.4.10. Amount of compensation . . . . . . . . 23",
        "4. PASSENGERS' RIGHTS",
        "4.4.7.", "'Long delays' at arrival", body,
        "4.4.10. Amount of compensation", "The compensation may be reduced by 50 % and amounts to EUR 300.",
        "5.1. Principle", "An air carrier is exempted only if it proves extraordinary circumstances.",
    ])
    secs = _split_into_sections(text, doc_id="g", doc_title="G", publisher="p", jurisdiction="EU",
                                airline_iata="", source_url="", effective_date=None)
    ids = [s.section_id for s in secs if s.section_id != "preamble"]
    assert ids == ["4-4-7-long-delays-at-arrival", "4-4-10-amount-of-compensation", "5-1-principle"]
    assert secs[[s.section_id for s in secs].index(ids[0])].text.startswith("Passengers reaching")


def _topic_chunk(cid, doc_id, breadcrumb, section="s"):
    return Chunk(chunk_id=cid, doc_id=doc_id, doc_title=breadcrumb.split(" > ")[0], publisher="p",
                 jurisdiction="EU", section_id=section, breadcrumb=breadcrumb, text="t", embed_text="t",
                 source_url="u")


def test_denied_boarding_sections_are_recognised_from_their_own_headings():
    from flight_delay.retrieval import is_denied_boarding_chunk

    assert is_denied_boarding_chunk(_topic_chunk("a", "us-cfr-250-oversales", "14 CFR Part 250 > § 250.1"))
    assert is_denied_boarding_chunk(_topic_chunk("b", "eu-261-2004", "Reg 261 > Article 4 Denied boarding"))
    assert is_denied_boarding_chunk(_topic_chunk(
        "c", "aa", "American > Conditions of Carriage > Your flight > Oversold flights > Voluntary"))
    # A heading that also covers delay or cancellation is not about bumping alone.
    assert not is_denied_boarding_chunk(_topic_chunk(
        "d", "g", "Guidelines > 4.4. Right to compensation in the event of denied boarding, cancellation, delay"))
    # A document title listing every topic does not count.
    assert not is_denied_boarding_chunk(_topic_chunk(
        "e", "caa", "CAA — Delays, Cancellations and Denied Boarding > Delays > Compensation"))


def test_section_cap_keeps_the_best_two_chunks_of_a_section():
    from flight_delay.retrieval import cap_per_section

    ranked = [_topic_chunk(f"r24-{i}", "ua", "UA > Rule 24", section="rule-24") for i in range(3)]
    ranked.insert(1, _topic_chunk("a7", "uk", "UK > Article 7", section="article-7"))
    kept = cap_per_section(ranked, 2)
    assert [c.chunk_id for c in kept] == ["r24-0", "a7", "r24-1"]
    assert cap_per_section(ranked, 0) == ranked


def test_retrieval_query_carries_the_flight_facts_the_question_lacks():
    """live-01: "UA 9901 is showing a big delay" searched without Heathrow, the UK or
    the arrival delay, and retrieved UK261 Art 6 without Art 7 (Session 21)."""
    from flight_delay.pipeline import plan_turn, retrieval_query

    live = FlightStatus(
        flight_iata="UA9901", airline_iata="UA", status="active", dep_iata="LHR", dep_time=None,
        dep_estimated=None, dep_delayed=300, arr_iata="ORD", arr_time=None, arr_estimated=None,
        arr_delayed=290, delayed=300,
    )
    question = "Flight UA 9901 is showing a big delay. What am I owed?"
    plan = plan_turn(question, [], lambda _no: live)
    q = retrieval_query(plan.question, plan, live)
    assert q.startswith(question)
    for fact in ("United Airlines", "LHR (United Kingdom)", "ORD (United States)", "delay",
                 "arrival delay 4 h 50 min", "UK261", "compensation for a long delay at arrival"):
        assert fact in q, fact
    # A general question with nothing to add searches with the words alone.
    general = plan_turn("What is a tarmac delay?", [], lambda _no: None)
    assert "law:" not in retrieval_query(general.question, general, None)


def test_only_a_bumping_question_lets_denied_boarding_text_in():
    from flight_delay.pipeline import disruptions_named, excluded_topics
    from flight_delay.retrieval import is_cancellation_only_chunk

    assert excluded_topics("I got bumped, what compensation?") == ()
    assert excluded_topics("My flight is 5 hours late") == ("denied_boarding", "cancellation")
    assert excluded_topics("My flight was cancelled") == ("denied_boarding",)
    assert is_cancellation_only_chunk(_topic_chunk("x", "caa", "CAA > Cancellations > Seven to 14 days' notice"))
    assert not is_cancellation_only_chunk(_topic_chunk("y", "ua", "UA > Rule 24 Flight Delays/Cancellations"))

    assert "denied boarding" in disruptions_named("I got bumped from an oversold United flight")
    assert "denied boarding" in disruptions_named("American denied me boarding in Dublin")
    assert "denied boarding" not in disruptions_named("My flight is delayed 4 hours, what compensation?")
    assert disruptions_named("Is it on time?", _flight("LHR", "ORD")) == ["cancellation"]


def test_a_cancelled_record_s_delay_minutes_are_not_a_delay():
    """Live UA169 (Session 37): 'cancelled, delay 235 min'. The minutes added four delay
    lanes that crowded the cancellation ones out of the seats."""
    from dataclasses import replace

    from flight_delay.pipeline import disruptions_named

    cancelled = replace(_flight("VCE", "EWR"), dep_delayed=235, delayed=235)
    assert disruptions_named("What compensation do I get?", cancelled) == ["cancellation"]
    # Said by the passenger, it still counts; and a delayed (not cancelled) record still counts.
    assert "delay" in disruptions_named("It was delayed and then cancelled", cancelled)
    delayed = replace(cancelled, status="active")
    assert disruptions_named("What compensation do I get?", delayed) == ["delay"]


def test_trigger_lane_seats_the_cancellation_article_for_eu_and_uk_only():
    from flight_delay.pipeline import plan_turn, search_lanes, trigger_lanes

    s = Settings(_env_file=None)
    eu = plan_turn("My United flight from Paris to Chicago was cancelled. What am I owed?",
                   [], lambda _no: None)
    lanes = trigger_lanes(eu.question, eu, None)
    assert [(j, k) for j, _q, k in lanes] == [("EU", "trigger")]
    # The lane names the legal concept, never the article or an amount.
    assert not re.search(r"article|\d|€|£", lanes[0][1].split(" ", 1)[1], re.I)
    # Handed out after scope and before remedy.
    kinds = [k for _j, _q, k in search_lanes(eu.question, eu, None, s)]
    assert kinds.index("trigger") > max(i for i, k in enumerate(kinds) if k == "scope")
    assert kinds.index("trigger") < kinds.index("remedy")
    uk = plan_turn("My Delta flight from London to Atlanta was cancelled. What am I owed?",
                   [], lambda _no: None)
    assert [j for j, _q, _k in trigger_lanes(uk.question, uk, None)] == ["UK"]
    # No trigger article under US law, none for a delay, none for a non-entitlement question.
    us = plan_turn("My American flight from Dallas to Chicago was cancelled. What am I owed?",
                   [], lambda _no: None)
    assert trigger_lanes(us.question, us, None) == ()
    late = plan_turn("My United flight from Paris to Chicago was delayed 5 hours. What am I owed?",
                     [], lambda _no: None)
    assert trigger_lanes(late.question, late, None) == ()
    assert trigger_lanes("Was my Paris flight cancelled?", eu, None) == ()


def test_pipeline_searches_with_the_structured_query_and_the_topic_filter():
    pipe, retriever = _pipeline()
    calls = []
    orig = retriever.retrieve

    def spy(query, **kw):
        calls.append((query, kw))
        return orig(query, **kw)

    retriever.retrieve = spy
    pipe.run("My Delta flight from Atlanta to Tampa is delayed 4 hours. What am I owed?")
    query, kw = calls[0]
    assert "Delta Air Lines flight from Atlanta (United States) to Tampa (United States)" in query
    assert kw["exclude_topics"] == ("denied_boarding", "cancellation")


# ==========================================================================
# Session 23: tarmac wording, complaint intent, remedy lanes
# ==========================================================================


def test_natural_tarmac_wording_is_a_tarmac_delay():
    from flight_delay.pipeline import disruptions_named

    q = ("Our Southwest flight sat on the ground at Chicago Midway for 2 hours after landing and "
         "we couldn't get off. What were they required to provide?")
    assert "tarmac delay" in disruptions_named(q)
    assert "tarmac delay" in disruptions_named("We were not allowed to get off the plane for 3 hours")
    assert "tarmac delay" not in disruptions_named("My flight was delayed 3 hours at the gate")


def test_a_complaint_question_is_searched_as_a_complaint_not_a_cancellation():
    """dom-44: "cancelled flight" used to steer the search to Delta's cancellation pages."""
    from flight_delay.pipeline import plan_turn, retrieval_query

    q = "Delta never responded to my written complaint about a cancelled flight. How long do they have?"
    plan = plan_turn(q, [], lambda _no: None, flt={"airline_scope": "DL", "jurisdiction_scope": ["US"],
                                                   "jurisdiction_governing": ["US"]})
    rq = retrieval_query(q, plan, None)
    assert "consumer complaint" in rq and "disruption: cancellation" not in rq


def test_remedy_lanes_name_remedies_per_governing_regime_never_articles():
    from flight_delay.pipeline import plan_turn, remedy_lanes

    live = FlightStatus(
        flight_iata="UA9901", airline_iata="UA", status="active", dep_iata="LHR", dep_time=None,
        dep_estimated=None, dep_delayed=300, arr_iata="ORD", arr_time=None, arr_estimated=None,
        arr_delayed=290, delayed=300,
    )
    q = "Flight UA 9901 is showing a big delay. What am I owed?"
    plan = plan_turn(q, [], lambda _no: live)
    lanes = remedy_lanes(plan.question, plan, live)
    uk = [text for juris, text in lanes if juris == "UK"]
    assert len(uk) == 3 and all(text.startswith("UK261 ") for text in uk)
    assert any("compensation amount" in t for t in uk) and any("care" in t for t in uk)
    assert not any(re.search(r"article|\d{2,4}|£|€", t, re.I) for _j, t in lanes)
    # A plain status question gets no lanes.
    status = plan_turn("Is UA 9901 delayed?", [], lambda _no: live)
    assert remedy_lanes(status.question, status, live) == ()


def test_lane_hits_join_the_pool_and_vote_in_the_final_order():
    from flight_delay.retrieval import HybridRetriever

    def ch(cid, juris, text, doc_type="regulation"):
        return Chunk(chunk_id=cid, doc_id=cid, doc_title=cid, publisher="p", jurisdiction=juris,
                     section_id=cid, breadcrumb=f"{cid} > s", text=text, embed_text=text,
                     source_url="u", doc_type=doc_type)

    store = MemoryStore()
    chunks = [ch("a6", "UK", "delay beyond scheduled departure assistance meals"),
              ch("a7", "UK", "passengers shall receive compensation amounting to 520 by distance"),
              ch("x", "UK", "denied boarding volunteers")]
    emb = HashEmbedder(64)
    store.upsert(chunks, emb.embed_documents([c.embed_text for c in chunks]))
    s = Settings(embedder="hash", reranker="none", candidates_k=1, final_k=1, sparse_candidates_k=0,
                 balance_sources=False, governing_candidates_k=0, lane_candidates_k=1,
                 fuse_rerank_with_dense=True)
    from flight_delay.retrieval import NoopReranker
    r = HybridRetriever(store, emb, NoopReranker(), s)
    base = r.retrieve("delay beyond scheduled departure", flt={"jurisdiction": "UK"})
    with_lane = r.retrieve("delay beyond scheduled departure", flt={"jurisdiction": "UK"},
                           lanes=(("UK", "compensation amounting to 520 by distance"),))
    # Dense top-1 is a6 either way; the compensation lane brings a7 into the pool.
    assert "a7" not in [c.chunk_id for c in base.ranked]
    assert "a7" in [c.chunk_id for c in with_lane.ranked]
    assert "lanes" in with_lane.timings_ms


def test_scope_lanes_cover_every_regime_in_scope_not_only_the_governing_ones():
    from flight_delay.pipeline import plan_turn, scope_lanes

    # US -> Paris: US governs, EU261 is only in scope - and EU is exactly the regime
    # whose scope the answer has to cite to say it does not apply (Rule 3).
    plan = plan_turn("My United flight from New York to Paris was cancelled. What am I owed?",
                     [], lambda _no: None)
    assert "EU" in plan.route.scope and "EU" not in (plan.route.governing or ())
    assert {j for j, _q, _k in scope_lanes(plan)} == set(plan.route.scope)
    assert all(kind == "scope" for _j, _q, kind in scope_lanes(plan))
    # The lanes name the legal concept, never the article that states it. (The text
    # after the regime's own name: the lane is labelled "EU261: ..." like a remedy lane.)
    bodies = [q.split(": ", 1)[1] for _j, q, _k in scope_lanes(plan)]
    assert not any(re.search(r"article|\d{2,4}|£|€", b, re.I) for b in bodies)
    # Nobody asks whether US law covers a US domestic flight.
    dom = plan_turn("My American flight from Dallas to Chicago was cancelled. What am I owed?",
                    [], lambda _no: None)
    assert tuple(dom.route.scope) == ("US",) and scope_lanes(dom) == ()
    # Nor about a general question with no route.
    gen = plan_turn("How much compensation for denied boarding?", [], lambda _no: None)
    assert scope_lanes(gen) == ()


def _toronto_plan():
    """The live Session 37 conversation: asked once, flight number not found."""
    from flight_delay.pipeline import plan_turn

    q = "My Toronto flight was cancelled today. what compensation do I get?"
    first = plan_turn(q, [], lambda _no: None)
    assert first.clarification
    return plan_turn("UA3874", [("user", first.question), ("assistant", first.clarification)],
                     lambda _no: None)


def test_a_route_from_outside_us_eu_uk_is_told_no_law_governs_and_gets_us_coverage_text():
    from flight_delay.pipeline import route_law_note, scope_lanes

    plan = _toronto_plan()
    assert plan.route.governing == () and plan.route.scope == ("US",)
    note = route_law_note(plan.route)
    assert note.startswith("LAW NOTE:") and "MAY apply" in note and "not in the sources" in note
    assert "government" in note and "airline" in note
    # The note reports routing, never a rule's content.
    assert not re.search(r"\d|€|£|\$|refund|compensat", note, re.I)
    # US law is only POSSIBLE here, so its coverage text must be there to cite.
    assert [(j, k) for j, _q, k in scope_lanes(plan)] == [("US", "scope")]


def test_montreal_to_newark_says_us_rules_may_apply_and_refers_the_passenger():
    """The live UA3623 YUL->EWR case: departure outside US/EU/UK, arrival in the US.
    User's rule (2026-09-22): not sure, DOT MAY apply, see that government's site."""
    from dataclasses import replace

    from flight_delay.pipeline import plan_turn, route_law_note, scope_lanes

    rec = replace(_flight("YUL", "EWR"), flight_iata="UA3623")
    plan = plan_turn("My boyfriend's Montreal flight is canceled. UA3623", [], lambda _no: rec)
    assert plan.clarification is None
    assert plan.route.governing == () and plan.route.scope == ("US",)
    note = route_law_note(plan.route)
    assert "MAY apply" in note and "government" in note and "airline" in note
    assert [(j, k) for j, _q, k in scope_lanes(plan)] == [("US", "scope")]
    # EU/UK departures are untouched: EU -> US still governs under both.
    eu = plan_turn("My United flight UA57", [], lambda _no: _flight("CDG", "EWR"))
    assert eu.route.governing == ("US", "EU") and route_law_note(eu.route) is None


def test_law_note_stays_silent_when_a_regime_governs_or_no_place_is_named():
    from flight_delay.pipeline import plan_turn, route_law_note

    for q in ("My United flight from London to Chicago was cancelled. What am I owed?",
              "My American flight from Dallas to Chicago was cancelled. What am I owed?",
              "How much compensation for denied boarding?"):
        assert route_law_note(plan_turn(q, [], lambda _no: None).route) is None, q


def test_the_law_note_reaches_the_model():
    llm = _ScriptedLLM("By law you are owed a refund [S1].")
    pipe = _run_pipeline(llm)
    pipe.run("My Toronto flight was cancelled today. what compensation do I get?",
             conversation_id="t1")
    pipe.run("UA3874", conversation_id="t1")
    assert llm.prompts and "LAW NOTE:" in llm.prompts[-1]


def test_procedural_lanes_fire_only_on_their_own_wording():
    from flight_delay.pipeline import procedural_lanes

    credit = procedural_lanes("United offered me a travel credit. Do I have to accept it?")
    assert len(credit) == 1 and "affirmatively accept" in credit[0][1]
    assert credit[0][0] == "US" and credit[0][2] == "procedural"
    assert any("prompt refund" in q for _j, q, _k in
               procedural_lanes("I paid cash for my ticket. When will I get my refund?"))
    assert any("smaller equipment" in q for _j, q, _k in
               procedural_lanes("United swapped to a smaller plane for operational reasons."))
    # A plain delay question asks for none of them.
    assert procedural_lanes("My flight was delayed four hours. What am I owed?") == ()


def test_airline_quota_drops_only_for_a_question_no_carrier_page_can_answer():
    from flight_delay.pipeline import airline_quota

    s = Settings(min_airline_sources=2, min_airline_sources_legal=1)
    assert airline_quota("How much does UK261 owe me for a five-hour delay?", s) == 1
    assert airline_quota("Does Delta owe me a hotel?", s) == 2
    assert airline_quota("What does American provide after a diversion?", s) == 2
    # Equal minimums reproduce the behaviour before the quota was made dynamic.
    flat = Settings(min_airline_sources=2, min_airline_sources_legal=2)
    assert airline_quota("How much does UK261 owe me for a five-hour delay?", flat) == 2


def test_a_scope_lane_leader_takes_a_seat_and_is_marked_guaranteed():
    from flight_delay.retrieval import HybridRetriever, NoopReranker

    def ch(cid, text):
        return Chunk(chunk_id=cid, doc_id=cid, doc_title=cid, publisher="p", jurisdiction="UK",
                     section_id=cid, breadcrumb=f"{cid} > s", text=text, embed_text=text,
                     source_url="u", doc_type="regulation")

    emb = HashEmbedder(64)
    base = dict(embedder="hash", reranker="none", candidates_k=2, final_k=2, sparse_candidates_k=0,
                balance_sources=True, min_government_sources=0, min_airline_sources=0,
                governing_candidates_k=0, lane_candidates_k=1, fuse_rerank_with_dense=True)
    q = "delay beyond the scheduled time of departure assistance and meals"

    def retriever(**over):
        # A store of its own each time: MemoryStore hands back the SAME Chunk objects
        # on every call, so a `guaranteed` set by one retrieval would outlive it here.
        store = MemoryStore()
        chunks = [ch("a6", q),
                  ch("a7", "compensation amounting to 520 by distance band"),
                  ch("a3", "this regulation applies to passengers departing from an airport")]
        store.upsert(chunks, emb.embed_documents([c.embed_text for c in chunks]))
        return HybridRetriever(store, emb, NoopReranker(), Settings(**base, **over))

    lanes = (("UK", "this regulation applies to passengers departing", "scope"),)
    final = retriever(lane_seats=True, max_lane_seats=1).retrieve(q, lanes=lanes).chunks
    a3 = next(c for c in final if c.chunk_id == "a3")
    # Marked, because it is here by the route and the question, not by relevance:
    # build_context must not treat it as the weakest thing present (Session 19).
    assert a3.guaranteed is True

    # A remedy lane votes and nothing more, unless seat_remedy_lanes says otherwise.
    remedy = (("UK", "this regulation applies to passengers departing", "remedy"),)
    off = retriever(lane_seats=True, max_lane_seats=1, seat_remedy_lanes=False)
    assert not any(c.guaranteed for c in off.retrieve(q, lanes=remedy).chunks)


def test_lane_seats_never_crowd_out_the_governing_guarantee():
    from flight_delay.retrieval import balance_by_source_class

    def ch(cid, juris):
        return Chunk(chunk_id=cid, doc_id=cid, doc_title=cid, publisher="p", jurisdiction=juris,
                     section_id=cid, breadcrumb=cid, text=cid, embed_text=cid, source_url="u",
                     doc_type="regulation")

    ranked = [ch("us1", "US"), ch("us2", "US"), ch("uk1", "UK"), ch("seat", "US")]
    out = balance_by_source_class(ranked, 3, min_government=1, min_airline=0,
                                  require_jurisdictions=("US", "UK"),
                                  prefer=[[ranked[-1]]])
    # The regime guarantee is filled first; the lane seat takes what is left.
    assert {c.chunk_id for c in out} == {"us1", "uk1", "seat"}


def test_a_lane_falls_through_to_its_next_candidate_rather_than_repeat_a_section():
    from flight_delay.retrieval import balance_by_source_class

    def ch(cid, section):
        return Chunk(chunk_id=cid, doc_id="d", doc_title="d", publisher="p", jurisdiction="UK",
                     section_id=section, breadcrumb=cid, text=cid, embed_text=cid, source_url="u",
                     doc_type="regulation")

    # The lane's best hit repeats the section the quota already took; its next does not.
    ranked = [ch("a5a", "art-5"), ch("a5b", "art-5"), ch("a8", "art-8")]
    lane = [ranked[1], ranked[2]]
    off = balance_by_source_class(ranked, 2, min_government=1, min_airline=0, prefer=[lane])
    assert [c.chunk_id for c in off] == ["a5a", "a5b"]
    on = balance_by_source_class(ranked, 2, min_government=1, min_airline=0, prefer=[lane],
                                 prefer_section_diverse=True)
    assert [c.chunk_id for c in on] == ["a5a", "a8"]


# ==========================================================================
# Session 27: false validation failures measured on the Session 26 subset
# (evals/runs/generation/20260920T014936.json). Half of the eight sentences
# reported as "uncited" carried a citation, or were a decline, and the model
# could not tell which sentence the count referred to. Each test below holds
# the REAL rejected sentence.
# ==========================================================================


def _ctx_eight_sources():
    return build_context([_chunk(f"c{i}", f"Rule {i}: the carrier owes the passenger care.")
                          for i in range(1, 9)], None)


def test_a_comma_joined_citation_is_read_as_one_marker_per_source():
    """live-01: the model wrote "[S5, S6]"; _MARKER_RE only knew "[S5][S6]", so two
    fully cited sentences were rejected as uncited. The meaning is unambiguous."""
    from flight_delay.generation import normalize_citation_markers

    assert normalize_citation_markers("meals and hotels [S5, S6].") == "meals and hotels [S5][S6]."
    assert normalize_citation_markers("[S5;S6][S1]") == "[S5][S6][S1]"
    # not a list of markers: left alone, so it is still an invalid citation
    assert normalize_citation_markers("[S5, see also]") == "[S5, see also]"

    report = validate_answer(
        "A delay of five hours or more means you may claim the maximum cash compensation "
        "of 520 pounds per passenger, and you can also choose to cancel and receive a full "
        "refund of the ticket price [S5, S6].",
        _ctx_eight_sources())
    assert report.ok, report.failures
    assert [c.marker for c in report.citations] == ["S5", "S6"]


def test_a_comma_joined_marker_naming_a_missing_source_is_still_caught():
    """Loosening the SYNTAX must not loosen the unknown-source check."""
    report = validate_answer("You are owed 600 euros [S1, S99].", _ctx_eight_sources())
    assert not report.ok and report.unknown_markers == ["S99"]


def test_a_sentence_ending_in_an_abbreviation_keeps_its_own_citation():
    """dom-24: "...file a complaint with the U.S. DOT [S8]." was split after "U.S.",
    leaving the claim in one sentence and its marker in the next."""
    report = validate_answer(
        "If you still do not receive the refund within 20 days, contact Southwest's "
        "customer service or file a complaint with the U.S. DOT [S8].",
        _ctx_eight_sources())
    assert report.ok, report.uncited_factual_sentences
    # a real sentence boundary is still a boundary
    two = validate_answer("You are owed 600 euros in cash compensation for this delay. "
                          "United must also provide meals and a hotel room overnight.",
                          _ctx_eight_sources())
    assert len(two.uncited_factual_sentences) == 2


def test_a_passive_source_coverage_decline_is_accepted():
    """eu-us-01: "The exact compensation amount is not specified in the sources you
    provided." is a Rule 7 decline, not a claim; every pattern was active-voice."""
    report = validate_answer(
        "The exact compensation amount is not specified in the sources you provided.",
        _ctx_eight_sources())
    assert report.ok and report.is_abstention
    # ...and it still may not carry a claim of its own
    joined = validate_answer(
        "The amount is not specified in the sources you provided, but you are owed 600 euros.",
        _ctx_eight_sources())
    assert not joined.ok and not joined.is_abstention


def test_a_policy_claim_is_not_excused_by_the_new_decline_wording():
    """"United does not cover hotels" names a carrier, not the sources."""
    report = validate_answer("United does not cover hotels or meals for this delay.",
                             _ctx_eight_sources())
    assert not report.ok and not report.is_abstention


def test_retry_note_quotes_the_sentences_that_were_rejected():
    """A count is not a target: nothing in "2 factual sentence(s) carry no citation"
    tells the model which two of its own sentences to fix."""
    from flight_delay.pipeline import RETRY_NOTE_MAX_CHARS, build_retry_note

    report = validate_answer(
        "You are entitled to a refund because Southwest cancelled the flight and you paid "
        "in cash. You are also owed 600 euros [S99].",
        _ctx_eight_sources())
    note = build_retry_note(report)
    assert "YOUR PREVIOUS ANSWER WAS REJECTED" in note
    assert "You are entitled to a refund because Southwest cancelled" in note
    assert "S99" in note                      # the invented marker is named too
    assert len(note) <= RETRY_NOTE_MAX_CHARS


def test_retry_note_stays_within_its_reserved_budget():
    """run() reserves RETRY_NOTE_MAX_CHARS per retry before sizing the evidence, so
    a note that outgrew the cap would silently overrun the request."""
    from flight_delay.generation import ValidationReport
    from flight_delay.pipeline import RETRY_NOTE_MAX_CHARS, build_retry_note

    report = ValidationReport(
        ok=False, failures=["40 factual sentence(s) carry no citation"], citations=[],
        unknown_markers=[f"S{i}" for i in range(20)],
        uncited_factual_sentences=[f"Sentence number {i} makes a claim about 600 euros "
                                   + "and rambles on " * 40 for i in range(40)],
        is_abstention=False)
    note = build_retry_note(report)
    assert len(note) <= RETRY_NOTE_MAX_CHARS
    assert "more)" in note                    # says how many it could not quote


def test_retry_note_on_an_unknown_marker_alone_still_says_what_to_do():
    from flight_delay.pipeline import build_retry_note

    report = validate_answer("You are owed 600 euros [S99].", _ctx_eight_sources())
    note = build_retry_note(report)
    assert "S99" in note and "Rewrite the whole answer" in note


# ==========================================================================
# Follow-up templates (no model call)
# ==========================================================================
# The risk these guard against is not "the reply reads badly". It is that a
# deterministic reply, written in Python, states law - which is exactly what the
# rest of this system refuses to do. So the first two tests below are about what
# the templates may SAY, and the rest about when they may fire at all.


# One grounded answer to follow up on: four cited sentences, four topics.
_PRIOR_ANSWER = (
    "American Airlines will rebook you on its next flight with available seats at no "
    "additional cost after a cancellation [S1].\n"
    "If you choose not to travel, you can request a refund of the unused ticket value [S2].\n"
    "American does not guarantee reimbursement for a hotel you book yourself without written "
    "authorisation [S1].\n"
    "A refund must be made within seven business days for a credit-card purchase [S2]."
)


class _TwoSourceRetriever(_FakeRetriever):
    def retrieve(self, query, *, top_k=None, candidates_k=None, flt=None, exclude_topics=(),
                 lanes=(), min_airline=None):
        from flight_delay.retrieval import RetrievalResult

        self.calls.append((query, flt))
        return RetrievalResult(
            chunks=[_chunk("aa1", "rebooking and hotels", airline="AA", doc_type="contract"),
                    _chunk("us1", "refunds")],
            timings_ms={}, n_dense=2, n_sparse=0, n_fused=2)


class _CountingLLM:
    """Always returns the same grounded answer, and counts how often it was asked."""

    def __init__(self, answer=_PRIOR_ANSWER):
        self.answer = answer
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return self.answer


def _followup_pipeline(llm=None, **settings):
    from flight_delay.pipeline import RagPipeline

    s = Settings(_env_file=None, embedder="hash", reranker="none",
                 **({"confidence_gate_enabled": False} | settings))
    return RagPipeline(_TwoSourceRetriever(), llm or _CountingLLM(), _FakeTool({}), s,
                       store=_StateStore())


def _answered_conversation(llm=None, **settings):
    """Turn one: a real, generated, validated answer. Returns (pipeline, llm)."""
    pipe = _followup_pipeline(llm, **settings)
    first = pipe.run("My American flight from Dallas to Chicago was cancelled. What am I owed?",
                     conversation_id="c1")
    assert first.outcome == "answered", first.outcome
    return pipe, pipe.llm


@pytest.mark.parametrize("spec", followups.TEMPLATES, ids=lambda s: s.id)
def test_a_follow_up_template_never_states_a_claim_of_its_own(spec):
    """
    The scaffold is written in Python, so nothing checks it against a source. It
    is therefore held to stating nothing that WOULD need one: no number, no
    amount, no obligation word. validate_answer is the arbiter, since it is the
    same judge the generated answers face.
    """
    from flight_delay.followups import NO_MATERIAL, SCAFFOLD_FORBIDDEN

    for sentence in (spec.lead, spec.step, NO_MATERIAL):
        text = sentence.format(airline="American Airlines")
        assert not SCAFFOLD_FORBIDDEN.search(text), f"{spec.id}: {text}"
        report = validate_answer(text, build_context([], None))
        assert report.uncited_factual_sentences == [], f"{spec.id}: {text}"


def test_a_follow_up_quotes_only_what_the_validator_already_accepted():
    """
    The safety property. A quoted sentence is reused on the same ground
    validate_answer accepted it on: a claim comes back with its marker, and a
    next step comes back as a next step. Nothing else is reusable.
    """
    from flight_delay.followups import reusable

    pipe, llm = _answered_conversation()
    reply = pipe.run("I want a refund instead.", conversation_id="c1")

    assert reply.outcome == "followup" and llm.calls == 1
    quoted = reply.diagnostics["followup"]["quoted"]
    assert quoted
    for sentence in quoted:
        assert sentence in _PRIOR_ANSWER          # verbatim, not paraphrased
        assert reusable(sentence)                 # cited claim, or accepted advice
        assert sentence in reply.text
    assert {c.marker for c in reply.citations} == set(reply.diagnostics["followup"]["markers"])


def test_an_uncited_next_step_is_reusable_but_an_uncited_claim_is_not():
    from flight_delay.followups import reusable

    assert reusable("A refund is owed within seven business days [S2].")
    assert reusable("If you prefer to travel on another flight, request rebooking on the "
                    "next available flight.")
    # A claim with no marker was never accepted as a claim, so it cannot come back.
    assert not reusable("American must rebook you on the next flight at no additional cost.")
    # Nor may advice carrying a figure: out of its turn there is nothing to check it against.
    assert not reusable("Ask them to refund you within 7 business days.")
    # Framing and sign-offs are not reusable either.
    assert not reusable("If you need more detail on the airline's rebooking policy, let me know.")


def test_the_reported_aa2123_rebooking_follow_up_reuses_the_advice_it_was_given():
    """
    Reported defect: the answer's rebooking advice lived in an UNCITED bullet
    (correctly - it is advice), the selector demanded a marker, so "can you put
    me on the next available flight?" found nothing and told the passenger to
    ask the whole question again.
    """
    delivered = (
        "By law you are entitled to a full refund of the fare and any ancillary fees for the "
        "cancelled flight [S1].\n\n"
        "**What to do next**\n"
        "- Ask the airline for the refund voucher or credit and confirm the refund date.\n"
        "- If you prefer to travel on another flight, request rebooking on the next available "
        "flight.\n"
    )
    pipe, llm = _answered_conversation(_CountingLLM(delivered))
    reply = pipe.run("Can you put me on the next available flight?", conversation_id="c1")

    assert reply.outcome == "followup" and llm.calls == 1
    assert reply.diagnostics["followup"]["had_material"] is True
    assert "request rebooking on the next available flight." in reply.text
    assert "did not cover" not in reply.text


def test_a_rebooking_follow_up_does_not_imply_this_assistant_can_book():
    """There is no booking integration; a passenger at a gate must not think there is."""
    pipe, _ = _answered_conversation()
    for question in ("Can you put me on the next available flight?",
                     "I still want to travel. What should I do?"):
        reply = pipe.run(question, conversation_id="c1")
        assert "cannot book anything for you" in reply.text, question


def test_a_follow_up_costs_no_model_call_however_many_are_asked():
    pipe, llm = _answered_conversation()
    for question in ("I still want to travel. What should I do?",
                     "Do they have to give me a hotel?",
                     "I already paid for a hotel. Can I get reimbursed?",
                     "How long does the airline have to refund me?",
                     "What documents or receipts should I keep?",
                     "How do I file a claim?"):
        answer = pipe.run(question, conversation_id="c1")
        assert answer.outcome == "followup", question
        assert answer.generation_attempted is False
    assert llm.calls == 1                      # the original answer, and nothing since


def test_a_different_disruption_is_a_new_question_not_a_follow_up():
    """Session 37, live: after a cancellation answer, "How much compensation for denied
    boarding?" matched a template and got the cancellation notice rule quoted back."""
    pipe, llm = _answered_conversation()
    assert pipe.store.states["c1"]["last_answer"]["disruptions"] == ["cancellation"]
    # The same disruption still gets the fast, quoted reply.
    same = pipe.run("I don't know why it was cancelled.", conversation_id="c1")
    assert same.outcome == "followup" and llm.calls == 1
    bumped = pipe.run("How much compensation for denied boarding?", conversation_id="c1")
    assert bumped.outcome != "followup" and bumped.generation_attempted
    assert llm.calls == 2
    # That answer is now the one follow-ups quote, and it is about denied boarding.
    assert pipe.store.states["c1"]["last_answer"]["disruptions"] == ["denied boarding"]


def test_an_answer_saved_before_the_disruptions_field_takes_the_full_path():
    pipe, llm = _answered_conversation()
    pipe.store.states["c1"]["last_answer"].pop("disruptions")
    assert pipe.run("I was bumped. What compensation?", conversation_id="c1").outcome != "followup"
    # No disruption named: still a follow-up.
    assert pipe.run("How do I file a claim?", conversation_id="c1").outcome == "followup"


def test_a_follow_up_names_the_carrier_the_conversation_settled_on():
    pipe, _ = _answered_conversation()
    reply = pipe.run("Can I fly with another airline instead?", conversation_id="c1")
    assert "American Airlines" in reply.text and "{airline}" not in reply.text


def test_a_follow_up_with_nothing_to_quote_invents_nothing():
    """
    The previous answer covered refunds and hotels, not the cause of the
    cancellation. The reply must say so and still give the next step, not produce
    a plausible sentence about weather.
    """
    pipe, llm = _answered_conversation()
    reply = pipe.run("They said it was weather.", conversation_id="c1")

    assert reply.outcome == "followup" and llm.calls == 1
    assert reply.diagnostics["followup"]["had_material"] is False
    assert reply.citations == [] and "[S" not in reply.text
    assert "did not cover that" in reply.text
    # ...and it still tells them what to do about it.
    assert "Ask American Airlines" in reply.text


def test_a_follow_up_never_asks_the_passenger_to_restate_their_situation():
    """
    They asked a follow-up precisely because the carrier, flight, route and
    disruption are already settled. Sending them back to name all four again was
    the worst thing this feature did (reported on a cancelled AA2123).
    """
    from flight_delay.followups import NO_MATERIAL

    text = NO_MATERIAL.format(airline="American Airlines").lower()
    for demand in ("question of its own", "naming the airline", "what went wrong",
                   "ask it as a", "start again"):
        assert demand not in text, demand


def test_followup_fallback_generate_pays_for_the_call_instead_of_declining():
    pipe, llm = _answered_conversation(followup_fallback="generate")
    reply = pipe.run("They said it was weather.", conversation_id="c1")
    assert reply.outcome == "answered" and llm.calls == 2


def test_a_run_of_follow_ups_all_quote_the_one_grounded_answer():
    """Follow-ups must not quote each other: the stored answer is carried forward."""
    pipe, _ = _answered_conversation()
    stored = pipe.store.states["c1"]["last_answer"]["text"]
    pipe.run("I want a refund instead.", conversation_id="c1")
    pipe.run("How long does the airline have to refund me?", conversation_id="c1")
    assert pipe.store.states["c1"]["last_answer"]["text"] == stored == _PRIOR_ANSWER


def test_the_first_turn_is_never_a_follow_up():
    """
    The same words that are a follow-up on turn two are a new question on turn
    one, because there is no answer to quote. Here they reach the clarifying
    question, which is what an opening "I want a refund instead" deserves.
    """
    pipe = _followup_pipeline()
    answer = pipe.run("I want a refund instead.", conversation_id="c1")
    assert answer.outcome == "clarified" and "followup" not in answer.diagnostics


def test_a_new_situation_is_not_answered_from_the_old_one():
    """A message naming a route AND a disruption is a new question, not a move."""
    pipe, llm = _answered_conversation()
    answer = pipe.run("My flight from London to Boston was delayed 5 hours.",
                      conversation_id="c1")
    assert answer.outcome != "followup" and llm.calls == 2


def test_a_flight_number_makes_it_a_new_question():
    """A flight the passenger has just typed is a new subject, not a follow-up."""
    pipe, llm = _answered_conversation()
    answer = pipe.run("Can you put me on the next available flight, AA100?",
                      conversation_id="c1")
    assert answer.outcome != "followup" and llm.calls > 1


def test_a_reply_to_a_clarifying_question_is_never_read_as_a_follow_up():
    """
    "I want a refund instead" answering "which airports?" would be a template
    reply built from an answer about a different situation. The last thing said
    being a clarifying question rules the whole path out.
    """
    from flight_delay.pipeline import CLARIFY_INTRO

    pipe, llm = _answered_conversation()
    pipe.store.msgs.append(("assistant", CLARIFY_INTRO + "\n1. Which airports?"))
    answer = pipe.run("I want a refund instead.", conversation_id="c1")
    assert answer.outcome != "followup" and llm.calls == 2


def test_a_long_message_is_a_new_question_however_it_reads():
    pipe, llm = _answered_conversation()
    long_message = ("I want a refund instead of the rebooking, and while I am at it "
                    "here is everything else that happened to me on this trip, "
                    "which I will now describe at considerable length. " * 2)
    assert len(long_message) > followups.FOLLOWUP_MAX_CHARS
    answer = pipe.run(long_message, conversation_id="c1")
    assert answer.outcome != "followup" and llm.calls == 2


def test_the_eval_harness_takes_the_full_path():
    """
    Both eval stages pass allow_followup=False (evals/golden/replay.py), so no
    golden case can be scored on a template reply instead of the pipeline.
    """
    pipe, llm = _answered_conversation()
    answer = pipe.run("I want a refund instead.", conversation_id="c1", allow_followup=False)
    assert answer.outcome == "answered" and llm.calls == 2


def test_follow_ups_can_be_switched_off_entirely():
    pipe, llm = _answered_conversation(followup_templates=False)
    answer = pipe.run("I want a refund instead.", conversation_id="c1")
    assert answer.outcome == "answered" and llm.calls == 2


def test_a_turn_that_produced_nothing_quotable_keeps_the_previous_answer():
    """A gated or rejected turn must not cost the conversation its follow-ups."""
    from flight_delay.pipeline import answer_record

    pipe, _ = _answered_conversation()
    stored = pipe.store.states["c1"]["last_answer"]
    pipe.llm = _ScriptedLLM("You are owed $1,550 in cash within 3 days.",
                            "You are owed $1,550 in cash within 3 days.")
    rejected = pipe.run("What about my connecting flight from Chicago that was cancelled?",
                        conversation_id="c1")
    assert rejected.outcome == "validation_failed"
    assert answer_record(rejected) is None
    assert pipe.store.states["c1"]["last_answer"] == stored


@pytest.mark.parametrize("question", [
    "I still want to travel. What should I do?",
    "Can you put me on the next available flight?",
    "Can I fly with another airline instead?",
    "I want a refund instead.",
    "I don't know why the flight was cancelled.",
    "They said it was weather.",
    "They said it was a mechanical issue.",
    "They said it was a crew problem.",
    "The airline offered me a flight tomorrow.",
    "The new flight gets me there the next day. What are my options?",
    "Do they have to give me a hotel?",
    "Do I get a meal voucher?",
    "Can I claim transportation to the hotel?",
    "I already paid for a hotel. Can I get reimbursed?",
    "I already accepted a travel credit. Can I still ask for a refund?",
    "I accepted the replacement flight. Can I still get compensation?",
    "How much compensation am I entitled to?",
    "Does it matter that I was flying from the UK?",
    "Does it matter that I was flying from the EU?",
    "What if the flight was from the US to Europe?",
    "What if I arrive 3 hours late?",
    "What if I arrive 5 hours late?",
    "What if the airline says the delay was outside its control?",
    "What if they don't tell me the reason?",
    "What if I miss my connection because of the cancellation?",
    "Can I choose not to travel anymore?",
    "What should I ask the gate agent?",
    "What documents or receipts should I keep?",
    "How do I file a claim?",
    "How long does the airline have to refund me?",
])
def test_every_listed_follow_up_is_recognised(question):
    """The thirty moves this feature was built for (see PROGRESS.md Session 31)."""
    assert followups.match(question) is not None


def test_a_specific_move_wins_over_the_general_one_that_contains_it():
    ids = {q: followups.match(q).id for q in (
        "I already paid for a hotel. Can I get reimbursed?",
        "Can I claim transportation to the hotel?",
        "Do they have to give me a hotel?")}
    assert ids == {"I already paid for a hotel. Can I get reimbursed?": "hotel_reimbursement",
                   "Can I claim transportation to the hotel?": "hotel_transport",
                   "Do they have to give me a hotel?": "hotel"}


def test_naming_the_route_is_not_by_itself_a_follow_up():
    """
    The route templates used to fire on any "from <place>", which both swallowed
    new questions and looped: their own next step asks for the airports.
    """
    assert followups.match("I flew from Paris to New York.") is None
    assert followups.match("My flight from London was cancelled.") is None
    assert followups.match("Does it matter that I was flying from the UK?").id == "from_uk"


def test_a_follow_up_reaches_the_passenger_without_its_markers(api_client):
    """The UI strips [Sn] from a template reply exactly as from a generated one."""
    pipe, _ = _answered_conversation()
    client = api_client(pipe)
    first = client.post("/ask", json={
        "question": "My American flight from Dallas to Chicago was cancelled. What am I owed?"})
    body = first.json()
    second = client.post("/ask", json={"question": "I want a refund instead.",
                                       "conversation_id": body["conversation_id"],
                                       "conversation_token": body["conversation_token"]})
    data = second.json()
    assert data["outcome"] == "followup" and data["citations"]
    stripped = re.sub(r"[ \t]*\[S\d+(?:\s*[,;]\s*S\d+)*\](?:[ \t]*\[S\d+(?:\s*[,;]\s*S\d+)*\])*",
                      "", data["answer"])
    assert "[S" not in stripped
