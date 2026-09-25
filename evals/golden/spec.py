"""
The shape of one golden case, and the vocabulary its tags may use.

Each case carries the four things a golden test case needs:
    1. question         the passenger's input
    2. refs             selectors for the exact chunks that answer it (context)
    3. expected         the reference answer (ground truth)
    4. tags             route / carrier / disruption type / controllability

Expected answers follow system_prompt.md: situation, the law (Tier 1), what the
airline adds (Tier 2/3), the cause caveat, next steps; under 300 words. They cite
by document and article (e.g. [UK261 Art 6(3)]) rather than [S1] markers, because
marker numbers depend on what a given retrieval run returns.
"""

from __future__ import annotations

from .refs import Ref

# Bumped for every deliberate change to what the golden set scores against.
# Evaluation results record it; results from different versions are not comparable.
#   1  gold context = the first CHUNK holding each phrase, at the configured size
#   2  (Session 12) gold passages resolved against parsed SECTIONS, independent of
#      chunk size (refs.py); parser fixes moved 13 cases' gold sections onto the
#      rules/sections that actually hold the clauses (PROGRESS.md, Session 12).
#      Questions, expected answers, facts and tags unchanged.
GOLDEN_VERSION = 2

CARRIERS = {
    "American": "AA", "Delta": "DL", "United": "UA", "Southwest": "WN", "Unspecified": "",
    # Not supported by the assistant: cases check that production declines or
    # answers from government rules only (tools.SUPPORTED_AIRLINES).
    "British Airways": "BA", "JetBlue": "B6",
}

DISRUPTIONS = {
    "Delay", "Cancellation", "Denied Boarding", "Tarmac Delay", "Missed Connection",
    "Diversion", "Schedule Change", "Downgrade", "Delayed Baggage", "Ancillary Fee",
    "Missed Flight", "Refund Procedure", "Complaint", "None",
}

CONTROLLABILITY = {
    "Controllable",      # the scenario states a carrier-side cause (mechanical, crew, IT)
    "Uncontrollable",    # the scenario states weather / ATC / strike
    "Unknown",           # cause not stated: the answer must be conditional
    "Passenger-caused",  # late arrival, missed check-in
    "N/A",               # the answer does not turn on cause (oversales, tarmac rules, refunds)
}

CATEGORIES = {
    "entitlement",           # what am I owed
    "route_applicability",   # which regime applies, or why one does not
    "ambiguous_direction",   # direction unknown -> clarifying question, then an assumed route
    "law_vs_airline",        # legal floor vs carrier promise, or a conflict between them
    "procedure",             # how/when: refund timing, claims, complaints
    "hybrid_live_data",      # answer depends on the live flight record
    "trap",                  # not answerable from the corpus -> abstain, or decline (unsupported airline)
}

# Edge / ambiguous cases. The builder requires at least EDGE_MIN_SHARE of all cases,
# and of the answerable ones (the only cases the retrieval evals score), to carry one.
EDGE_MIN_SHARE = 0.20
EDGE_KINDS = {
    # derived by the builder, never set by hand
    "ambiguous_route",       # production asks a clarifying question before answering
    "out_of_scope",          # trap: the corpus cannot answer it
    "passenger_caused",      # control="Passenger-caused"
    "unsupported_airline",   # production detects a carrier outside AA/DL/UA/WN
    # set with case(edge=...)
    "inapplicable_regime",   # the passenger invokes a regime that does not cover the trip
    "below_threshold",       # the delay is just short of the point where a right starts
    "multi_leg",             # connections or round trips: which leg decides
    "exception_clause",      # a notice period, award ticket or purchase location changes the answer
    "us_territory",          # a US territory airport (Guam, Puerto Rico...) that counts as US
    "other_region",          # an endpoint outside US/EU/UK: no regime may be inferred from it
}
EDGE_DERIVED = {"ambiguous_route", "out_of_scope", "passenger_caused", "unsupported_airline"}

# ---------------------------------------------------------------------------
# THE CANONICAL GOLDEN SET: exactly these 50 cases, in this order.
#
# Chosen by hand from the 121-case set (2026-09-14), not sampled, to keep in 50
# questions every production behaviour and retrieval challenge the old set
# exercised: the applicability matrix in both directions, clarification and
# assumption, scope vs governing, law vs airline promise, live flight data,
# US territories, OTHER-region routes and unsupported airlines. The builder fails
# unless the case files define exactly these ids; evals/chunk_sweep.py refuses a
# golden file that differs. The dropped cases are in retired_cases.py (unused).
#
#   id: (difficulty, retrieval stress)
#
# difficulty   easy    one clause or a routing decision with an obvious answer
#              medium  two documents, or one rule with a condition to apply
#              hard    several clauses, a conflict, a threshold/exception, or a
#                      routing trap that previously produced a wrong answer
# stress       what makes the case hard for RETRIEVAL (STRESS_KINDS)
# ---------------------------------------------------------------------------
GOLDEN_SIZE = 50
DIFFICULTIES = ("easy", "medium", "hard")
STRESS_KINDS = {
    "specific_clause",       # the answer turns on one precise provision
    "multi_chunk",           # the answer needs several chunks / documents
    "cross_jurisdiction",    # near-identical rules in US DOT / EU261 / UK261 must not be swapped
    "law_vs_airline",        # regulation and airline document both needed, and kept apart
    "unsupported_airline",   # government rules only; no AA/DL/UA/WN document may be used
    "chunk_boundary",        # the gold text sits in a long list/table a chunk boundary can split
}
CANONICAL_CASES: dict[str, tuple[str, tuple[str, ...]]] = {
    # -- US domestic: delay, cancellation, refund, denied boarding, tarmac, connection
    "dom-01": ("medium", ("multi_chunk", "law_vs_airline")),
    "dom-04": ("medium", ("specific_clause", "law_vs_airline")),
    "dom-07": ("medium", ("law_vs_airline",)),
    "dom-13": ("easy", ("specific_clause",)),
    "dom-15": ("medium", ("multi_chunk", "law_vs_airline")),
    "dom-17": ("medium", ("multi_chunk", "specific_clause")),
    "dom-21": ("hard", ("specific_clause", "law_vs_airline")),
    "dom-24": ("medium", ("specific_clause",)),
    "dom-25": ("medium", ("specific_clause", "chunk_boundary")),
    "dom-29": ("hard", ("specific_clause", "chunk_boundary")),
    "dom-31": ("medium", ("specific_clause",)),
    "dom-34": ("medium", ("specific_clause",)),
    "dom-36": ("hard", ("specific_clause", "chunk_boundary")),
    "dom-37": ("medium", ("law_vs_airline",)),
    "dom-44": ("easy", ("specific_clause",)),
    # -- US <-> Europe, both directions, and intra-Europe
    "uk-us-01": ("hard", ("cross_jurisdiction", "multi_chunk", "law_vs_airline")),
    "uk-us-03": ("hard", ("cross_jurisdiction", "law_vs_airline")),
    "uk-us-06": ("medium", ("specific_clause",)),
    "uk-us-09": ("hard", ("specific_clause", "cross_jurisdiction")),
    "uk-us-14": ("medium", ("specific_clause",)),
    "eu-us-01": ("hard", ("cross_jurisdiction", "multi_chunk")),
    "eu-us-05": ("hard", ("specific_clause", "multi_chunk")),
    "eu-us-07": ("medium", ("law_vs_airline", "cross_jurisdiction")),
    "eu-us-12": ("easy", ("specific_clause",)),
    "eu-us-13": ("hard", ("multi_chunk", "chunk_boundary")),
    "us-eur-01": ("medium", ("cross_jurisdiction",)),
    "us-eur-02": ("medium", ("cross_jurisdiction",)),
    "us-eur-04": ("hard", ("cross_jurisdiction", "chunk_boundary")),
    "us-eur-06": ("medium", ("specific_clause", "chunk_boundary")),
    "us-eur-09": ("hard", ("cross_jurisdiction",)),
    "intra-04": ("medium", ("cross_jurisdiction", "specific_clause")),
    "intra-07": ("medium", ("cross_jurisdiction",)),
    # -- ambiguous or incomplete route (clarify, then answer on an assumption)
    "amb-01": ("medium", ("cross_jurisdiction",)),
    "amb-02": ("medium", ("cross_jurisdiction",)),
    "amb-03": ("hard", ("cross_jurisdiction",)),
    "amb-06": ("medium", ("cross_jurisdiction",)),
    # -- law vs airline, and a general cross-regime question
    "law-02": ("hard", ("law_vs_airline", "multi_chunk")),
    "law-03": ("hard", ("law_vs_airline", "cross_jurisdiction")),
    "law-07": ("medium", ("cross_jurisdiction", "multi_chunk")),
    # -- hybrid: synthetic flight data + rights
    "live-01": ("medium", ("cross_jurisdiction",)),
    "live-02": ("medium", ("cross_jurisdiction",)),
    "live-03": ("easy", ("law_vs_airline",)),
    "live-04": ("easy", ("law_vs_airline",)),
    "live-05": ("hard", ("specific_clause",)),
    "live-06": ("medium", ("cross_jurisdiction",)),
    # -- edge: OTHER region, traps, unsupported airlines
    "other-01": ("hard", ("cross_jurisdiction",)),
    "trap-04": ("easy", ()),
    "trap-08": ("easy", ("unsupported_airline",)),
    "unsup-01": ("medium", ("unsupported_airline", "cross_jurisdiction")),
    "unsup-02": ("easy", ("unsupported_airline",)),
}
assert len(CANONICAL_CASES) == GOLDEN_SIZE

# The Sturgeon gap (EU261 delay compensation from CJEU case law, flagged on four
# cases through Session 21) is closed: the Commission's 2024 interpretative
# guidelines (doc eu-261-guidelines) state it, and those cases cite them (Session 22).
# `gap=` stays available for the next right the corpus cannot support.


def case(
    id: str,
    question: str,
    refs: list[Ref],
    expected: str,
    *,
    route: str,
    carrier: str,
    disruption: str,
    control: str,
    category: str = "entitlement",
    origin: str | None = None,
    dest: str | None = None,
    gap: str | None = None,
    notes: str | None = None,
    answerable: bool = True,
    clarify: bool = False,
    fixture: dict | None = None,
    status: str = "draft",
    reply: str | None = None,
    edge: str | None = None,
    expected_facts: list[str] | None = None,
    forbidden_claims: list[str] | None = None,
) -> dict:
    return {
        "id": id,
        "question": " ".join(question.split()),
        "refs": refs,
        "expected": _paragraphs(expected),
        "route": route,
        "carrier": carrier,
        "disruption": disruption,
        "control": control,
        "category": category,
        "origin": origin,
        "dest": dest,
        "gap": gap,
        "notes": notes,
        "answerable": answerable,
        # The case STOPS at the clarifying question: the missing detail decides
        # which regime governs, so there is no correct final answer to score and
        # expected_output is production's own question. A clarify case must be
        # answerable=False (nothing to retrieve evidence for yet); `reply` is
        # still kept, so the builder can check the route the reply settles on.
        "clarify": clarify,
        "fixture": fixture,
        "status": status,
        # The passenger's answer if production asks a clarifying question first.
        # Default (golden/replay.py): the case's airports, else "I don't know".
        "reply": reply,
        "edge": edge,
        # The key facts a correct answer must contain, and the materially wrong
        # or unsafe claims it must not make. Used for behavior-based scoring
        # instead of exact-string matching against `expected`/expected_output.
        "expected_facts": expected_facts or [],
        "forbidden_claims": forbidden_claims or [],
    }


def _paragraphs(text: str) -> str:
    """Keep blank-line paragraph breaks, collapse wrapping inside paragraphs."""
    return "\n\n".join(" ".join(p.split()) for p in text.strip().split("\n\n"))
