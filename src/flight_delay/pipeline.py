"""
Orchestration: route -> gather evidence -> generate -> validate -> maybe retry.

WHY A HAND-WRITTEN STATE MACHINE RATHER THAN LangGraph, FOR NOW
----------------------------------------------------------------
The full plan specifies LangGraph, and for a long-lived project that is right:
you get checkpointing, streaming of intermediate state, and a visualisable
graph. Under a two-day deadline it is the wrong trade. LangGraph adds a
dependency whose API has changed repeatedly, and debugging a framework you are
learning at the same time as the domain is exactly the failure mode we are
trying to avoid.

This module implements the same graph explicitly. It is ~150 lines, has no
framework magic, and every edge is visible. `run()` is a direct translation of
the graph in ARCHITECTURE.md:

    classify -> {policy | status | hybrid} -> assemble -> generate -> validate
                                                             ^          |
                                                             +--retry---+

Porting it to LangGraph later is mechanical: each function b
WHY THE ROUTER IS RULES-BASED AND NOT AN LLM CALL
--------------------------------------------------
An LLM classifier would add a full round-trip (~1s) and a failure mode to every
single request, to decide between three branches that are cleanly separable by
"does the question contain a flight number" and "does it ask about rules". The
rules version is instant, deterministic, free, and testable. If the eval shows
routing accuracy is the bottleneck, upgrade it then - but measure first.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta

from . import followups
from .confidence import ConfidenceGate, abstention_answer
from .generation import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    LLMError,
    ValidationReport,
    build_context,
    build_user_prompt,
    drop_uncited_list_items,
    evidence_token_budget,
    sanitize_error_text,
    validate_answer,
)
from .jurisdiction import (
    AMBIGUOUS_PLACES,
    REGIME_NAMES,
    RouteJurisdiction,
    assume_route,
    detect_jurisdictions,
    display_name,
    region_of,
)
from .metrics import (
    airlabs_calls_total,
    airlabs_quota_remaining,
    rag_abstentions_total,
    rag_candidates,
    rag_confidence,
    rag_context_tokens,
    rag_end_to_end_seconds,
    rag_followups_total,
    rag_gate_abstentions_total,
    rag_requests_total,
    rag_rerank_top_score,
    rag_retries_total,
    rag_stage_duration_seconds,
    rag_validation_failures_total,
    rag_zero_result_total,
)
from .models import Answer, Citation, FlightStatus, Intent
from .tools import (
    AIRLINE_NAMES,
    SUPPORTED_AIRLINES,
    QuotaExceeded,
    airline_of_flight,
    extract_flight_number,
)

# The rejection note appended on a validation retry is capped, so the request
# budget can reserve a known amount for it (see evidence_token_budget). The
# reserve is subtracted whether or not a retry happens, so a note SHORTER than
# the cap buys the evidence nothing: length up to the cap is free.
RETRY_NOTE_MAX_CHARS = 1200
# How much of one rejected sentence is quoted back. Enough to identify it in the
# model's own text, short enough that several still fit the cap.
RETRY_QUOTE_MAX_CHARS = 200


def build_retry_note(report: ValidationReport) -> str:
    """
    What the model is told after `validate_answer` rejected its answer.

    It quotes the REJECTED SENTENCES back verbatim. "2 factual sentence(s) carry
    no citation" is a count, not a target: nothing in the answer tells the model
    which two of its sentences a deterministic validator read as claims, and it
    cannot re-derive the rule from the number. Measured on the Session 26 subset,
    every case that failed validation once failed its retry too, and four of the
    eight sentences involved were one marker away from passing.

    The note never says what to claim, only which sentence to cite or drop, so it
    cannot smuggle a fact the sources do not carry. It stays within
    RETRY_NOTE_MAX_CHARS, which run() reserves up front.
    """
    head = ("\n\nYOUR PREVIOUS ANSWER WAS REJECTED: "
            + "; ".join(report.failures)[:RETRY_NOTE_MAX_CHARS // 3] + ".")
    tail = ("\nRewrite the whole answer. Use only the markers listed above, one source per "
            "bracket. If the sources do not answer the question, say so explicitly and "
            "claim nothing else.")
    parts: list[str] = []
    if report.unknown_markers:
        parts.append("\nThese markers name no source I gave you: "
                     + ", ".join(report.unknown_markers) + ".")
    if report.uncited_factual_sentences:
        header = "\nThese sentences were read as factual claims carrying no citation:"
        howto = ("\nFor EACH sentence above: either end it with the marker of a supplied source "
                 "that actually says it, or delete the sentence. Do not add a new claim.")
        room = RETRY_NOTE_MAX_CHARS - len(head) - len(tail) - len(header) - len(howto) - len(
            "".join(parts)) - 24  # 24: the "(and N more)" line, if it is needed
        quoted: list[str] = []
        for n, sentence in enumerate(report.uncited_factual_sentences, 1):
            short = sentence if len(sentence) <= RETRY_QUOTE_MAX_CHARS else (
                sentence[:RETRY_QUOTE_MAX_CHARS - 3].rstrip() + "...")
            line = f'\n{n}. "{short}"'
            if len(line) > room:
                break
            quoted.append(line)
            room -= len(line)
        if quoted:
            dropped = len(report.uncited_factual_sentences) - len(quoted)
            parts.append(header + "".join(quoted)
                         + (f"\n(and {dropped} more)" if dropped else "") + howto)
    return head + "".join(parts) + tail

# Shown INSTEAD of an answer that still failed validation after its retries. It
# follows the system prompt's Rule 7 (say what cannot be determined, point to the
# right resource) and states no right of its own. The rejected text is kept in
# Answer.diagnostics for logs and evaluation, never shown.
VALIDATION_FAILURE_MESSAGE = (
    "I couldn't put together an answer I can fully back with citations from the "
    "regulations and airline policies I have, so I won't guess.\n\n"
    "What I can suggest:\n"
    "- Ask the airline's customer service to confirm what you are owed, and to cite the rule "
    "they rely on\n"
    "- Check the airline's Contract of Carriage and Customer Service Plan\n"
    "- For a disputed claim, the US DOT complaint process, or the national enforcement body "
    "for a flight that departed the EU or UK\n\n"
    "This is not legal advice."
)

# Shown when the model could not be reached or returned nothing usable.
LLM_ERROR_MESSAGE = (
    "I can't answer right now because the answer service is unavailable. Please try again in "
    "a moment. Nothing about your rights has been decided by this message."
)

# Bounded label values for rag_gate_abstentions_total (never free text).
_GATE_REASON_LABELS = (
    ("no chunks", "no_results"), ("rerank score", "low_rerank_score"),
    ("could not discriminate", "flat_margin"), ("question terms", "weak_grounding"),
)


def gate_reason_label(reason: str | None) -> str:
    for needle, label in _GATE_REASON_LABELS:
        if reason and needle in reason:
            return label
    return "other"


def _term_matcher(*terms: str) -> re.Pattern[str]:
    """
    One case-insensitive regex from whole-word terms.

    Each term spells out its own inflections (`refunds?`), because a bare stem
    between `\\b`s never matches an inflected word: `\\bcompensat\\b` misses
    "compensation". A space in a term matches any run of whitespace.
    """
    body = "|".join(t.replace(" ", r"\s+") for t in terms)
    return re.compile(rf"\b(?:{body})\b", re.I)


# Words that indicate the user wants a RULE, not a flight's current state.
# A delay alone is not one: "Is UA2402 delayed?" is a status question.
_POLICY_HINT = _term_matcher(
    r"entitle(?:d|ment|ments)?", r"compensat(?:e|ed|es|ing|ion|ions)",
    r"refund(?:s|ed|ing|able)?", r"owe[ds]?", r"rights?", r"rules?", r"polic(?:y|ies)",
    r"allowed", r"liable", r"liability", r"claim(?:s|ed|ing)?",
    r"reimburse(?:d|s|ment|ments)?", r"hotels?", r"meals?", r"baggage", r"bags?",
    r"over-?book(?:ed|ing)?", r"bumped", r"denied boarding", r"deny boarding",
    r"cancel(?:s|ed|led|ing|ling|lation|lations)?", r"ca(?:nc|n|c)e?l+(?:ed|ing|ation|ations)?",
    r"missed (?:(?:my|our|the|a) )?connect(?:ion|ions|ing flights?)",
    r"connect(?:ion|ing flight)s? (?:was|were|got|has been|have been) missed",
    r"regulations?", r"contracts?", r"carriage", r"can i", r"am i", r"must", r"should",
    # "What are my options?" / "do I get anything?" ask for rights, not status
    # (live-02, live-03 got only the 3-chunk status fallback before Session 17).
    r"options?", r"do (?:i|we) get", r"get anything", r"what (?:do|can) (?:i|we) do",
)
# Words that indicate the user wants live STATUS.
_STATUS_HINT = _term_matcher(
    r"delay(?:s|ed|ing)?", r"late", r"on time", r"status", r"cancel(?:ed|led)",
    r"departed", r"landed", r"arriv(?:e|ed|es|ing|al|als)", r"gates?", r"where is",
    r"track(?:ing)?",
)


# The four carriers in the corpus, by the names passengers actually type.
# Order matters: "american airlines" must be tried before "american".
_AIRLINE_PATTERNS = [
    ("AA", re.compile(r"\b(american airlines|american|aa)\b", re.I)),
    ("DL", re.compile(r"\b(delta air lines|delta|dl)\b", re.I)),
    # "United" must not match "United Kingdom" or "United States" — otherwise a
    # question about a UK flight on another carrier gets scoped to United.
    ("UA", re.compile(r"\b(united airlines|united(?!\s+(kingdom|states|arab))|ua)\b", re.I)),
    ("WN", re.compile(r"\b(southwest airlines|southwest|swa|wn)\b", re.I)),
]
# Carriers outside the corpus, by name, so "my British Airways flight" is not
# answered from American, Delta, United and Southwest documents. Only names that
# are not also everyday words or places ("Alaska", "Spirit", "Virgin").
_OTHER_AIRLINE_PATTERNS = [
    ("BA", re.compile(r"\bbritish airways\b", re.I)),
    ("VS", re.compile(r"\bvirgin atlantic\b", re.I)),
    ("EI", re.compile(r"\baer lingus\b", re.I)),
    ("AF", re.compile(r"\bair france\b", re.I)),
    ("KL", re.compile(r"\bklm\b", re.I)),
    ("LH", re.compile(r"\blufthansa\b", re.I)),
    ("IB", re.compile(r"\biberia\b", re.I)),
    ("FR", re.compile(r"\bryanair\b", re.I)),
    ("U2", re.compile(r"\beasyjet\b", re.I)),
    ("B6", re.compile(r"\bjet ?blue\b", re.I)),
    ("AS", re.compile(r"\balaska airlines\b", re.I)),
    ("NK", re.compile(r"\bspirit airlines\b", re.I)),
    ("F9", re.compile(r"\bfrontier airlines\b", re.I)),
    ("HA", re.compile(r"\bhawaiian airlines\b", re.I)),
    ("AC", re.compile(r"\bair canada\b", re.I)),
    ("WS", re.compile(r"\bwestjet\b", re.I)),
    ("TK", re.compile(r"\bturkish airlines\b", re.I)),
    ("EK", re.compile(r"(?<!arab )\bemirates\b", re.I)),
    ("QR", re.compile(r"\bqatar airways\b", re.I)),
]


def resolve_airline(question: str, flight_no: str | None = None) -> str | None:
    """
    The IATA code of the carrier the question is about, SUPPORTED OR NOT.

    A flight number with a real airline prefix is the strongest signal ("BA117"
    is British Airways even if the passenger also mentions American). A prefix
    that is no known airline ("B12" from "gate B12") yields to a name in the text,
    and is returned only when nothing else names the carrier.
    """
    code = airline_of_flight(flight_no)
    if code in AIRLINE_NAMES:
        return code
    for iata, pattern in (*_AIRLINE_PATTERNS, *_OTHER_AIRLINE_PATTERNS):
        if pattern.search(question):
            return iata
    return code


def detect_airline(question: str, flight_no: str | None = None) -> str | None:
    """
    Work out which SUPPORTED carrier the question is about, from its name or its
    flight number prefix. None for no carrier, or one this assistant does not cover.

    This matters more than it sounds. Without it, a question about United can
    retrieve Delta's policy and present it as United's — a confidently wrong
    answer about a specific company's obligations to a specific passenger. The
    flight number is the stronger signal and wins when both are present.
    """
    code = resolve_airline(question, flight_no)
    return code if code in SUPPORTED_AIRLINES else None


def route_filters(
    question: str,
    flight_no: str | None,
    live,
    flt: dict | None = None,
    *,
    reply_start: int | None = None,
):
    """
    Build the retrieval filters for one question. Returns (flt, route).

    Shared by the pipeline and the eval harnesses, so an evaluation measures
    retrieval under the same airline/route scoping that production uses.
    """
    # Scope to the airline asked about, when we can tell. This narrows the
    # airline half of the evidence without touching the regulation half —
    # see airline_scope in store._filter_sql. An explicit caller-supplied
    # filter wins, since the API exposes it for exactly that purpose.
    #
    # An unsupported carrier is scoped to ITS OWN code too: "this airline's
    # documents plus all regulations" is then regulations only, because the
    # corpus holds no documents for it. Leaving it unscoped would mix American,
    # Delta, United and Southwest policies into an answer about British Airways.
    airline = resolve_airline(question, flight_no)
    if airline and (flt is None or "airline_scope" not in flt):
        flt = {**(flt or {}), "airline_scope": airline}

    # Scope the regulation half by route. Live flight data gives real
    # airport codes, which beat anything we can infer from the question.
    route = detect_jurisdictions(
        question,
        dep_iata=live.dep_iata if live else None,
        arr_iata=live.arr_iata if live else None,
        reply_start=reply_start,
    )
    return _with_route(flt, route), route


def _with_route(flt: dict | None, route) -> dict:
    """
    Two different lists, deliberately: everything the route touches is
    RETRIEVABLE, but only the regime whose territory the flight departed
    is GUARANTEED a slot. A US->Paris flight should be able to cite
    EU261's scope article to explain why it does not apply, without
    EU261's compensation table being forced into the context.
    """
    if flt is not None and "jurisdiction_scope" in flt:
        return flt
    return {
        **(flt or {}),
        "jurisdiction_scope": list(route.scope),
        "jurisdiction_governing": list(route.governing),
    }


# --------------------------------------------------------------------------
# Clarification. The departure airport decides EU261/UK261, and the
# destination decides the US domestic/international thresholds, so a
# passenger asking about their own disrupted trip is asked for what is missing
# before being answered: one message, one to four numbered questions.
#
# One round only. If the reply still leaves the route open, the answer opens
# with the assumption it was built on ("Assuming you're flying from
# Manchester, UK.") so the passenger can correct it.
#
# evals/build_golden_set.py replays turns through plan_turn(), so the golden
# set's expected clarifications and assumptions are produced by this code.
# --------------------------------------------------------------------------

CLARIFY_INTRO = (
    "Sorry to hear that. To tell you what you're owed I need to know the route, because the "
    "rules depend on where the flight left from: flights leaving the EU or UK can get fixed "
    "cash compensation, while the same route flown the other way does not."
)


def clarification_message(route, flight_no: str | None, found_without_route: bool = False) -> str:
    """The clarifying questions for this route: only the ones still open."""
    lead = ""
    items: list[str] = []
    if flight_no and found_without_route and route.unknown_airport_codes:
        codes = " and ".join(route.unknown_airport_codes)
        lead = (f" I found flight {flight_no}, but I can't place airport code {codes} from its "
                "flight data.")
    elif flight_no and found_without_route:
        lead = (f" I found flight {flight_no}, but the flight data doesn't say which airports "
                "it flies between.")
    elif flight_no:
        lead = f" I couldn't find flight {flight_no} in the live flight data."
    else:
        items.append("What is your flight number (for example, DL123)? I'll look it up.")

    for name in route.ambiguous_places:
        uk, other = AMBIGUOUS_PLACES[name.lower()]
        items.append(f"When you say {display_name(name)}, do you mean {uk} or {other}?")
    for name in route.unknown_places:
        items.append(f'Which country are you flying from? I don\'t recognise "{name}".')

    origin_ambiguous = route.origin_place is not None and route.origin_place[1] == "AMBIGUOUS"
    dep_missing = route.origin_region is None and not origin_ambiguous
    arr_missing = route.destination_place is None
    if dep_missing and arr_missing:
        items.append("Which airport did you fly from, and which airport were you flying to?")
    elif dep_missing:
        items.append("Which airport did you fly from?")
    elif arr_missing and route.origin_region not in (None, "US"):
        # Leaving Europe (or elsewhere), only a US arrival brings in US DOT.
        items.append("Which airport were you flying to?")

    if len(items) == 1:
        body = items[0]
    else:
        body = "\n".join(f"{i}. {q}" for i, q in enumerate(items, start=1))
    return f"{CLARIFY_INTRO}{lead}\n\n{body}"


def is_clarification(text: str) -> bool:
    return text.startswith(CLARIFY_INTRO)


# --------------------------------------------------------------------------
# Unsupported airlines. The corpus holds American, Delta, United and Southwest
# documents only. A flight on any other carrier is recognised (so it is not
# mistaken for "no flight number"), but it is never looked up in AirLabs and no
# airline document is retrieved for it. The passenger gets either a plain
# "not covered" reply, or, when the question is about rights and the route is
# settled, an answer from government regulations alone, opened with a notice.
# --------------------------------------------------------------------------

UNSUPPORTED_INTRO = "I can only help with flights on American Airlines, Delta, United and Southwest."


def _unsupported_subject(code: str, flight_no: str | None) -> tuple[str, str]:
    """('BA117 is a British Airways flight', 'British Airways')."""
    name = AIRLINE_NAMES.get(code)
    if flight_no:
        subject = f"{flight_no} is a flight on {name}" if name else f"{flight_no} is not one of their flights"
    else:
        subject = f"{name} is not one of them" if name else "That airline is not one of them"
    return subject, name or "that airline"


def unsupported_airline_message(code: str, flight_no: str | None) -> str:
    """The reply when an unsupported carrier's question cannot be answered from the law alone."""
    subject, name = _unsupported_subject(code, flight_no)
    example = (f"My {flight_no or name} flight from London Heathrow to New York JFK was "
               "delayed 4 hours. What are my rights?")
    return (
        f"{UNSUPPORTED_INTRO} {subject}, so I can't look up the flight or tell you what {name} "
        "itself offers.\n\n"
        "Government passenger-rights rules (US DOT, EU261 or UK261) may still apply, depending on "
        "which airport the flight left from and where it was going. If you'd like those, ask again "
        f'with both airports, for example: "{example}"'
    )


def unsupported_airline_notice(code: str, flight_no: str | None, route: RouteJurisdiction) -> str:
    """The sentence a regulations-only answer for an unsupported carrier opens with."""
    subject, name = _unsupported_subject(code, flight_no)
    notice = (f"{UNSUPPORTED_INTRO} {subject}, so this answer covers only government "
              f"passenger-rights rules, not the policies of {name}.")
    # The departure rules (EU261/UK261 Art 3(1)(a)) bind every airline. A flight
    # INTO the EU/UK is covered only on an EU/UK-licensed carrier (Art 3(1)(b)):
    # carrier-specific, so it is flagged, not decided.
    dest = route.destination_region
    if dest in ("EU", "UK") and dest != route.origin_region:
        notice += (f" Whether {REGIME_NAMES[dest]} also covers a flight arriving in the {dest} "
                   "depends on where the airline is licensed, which I can't confirm.")
    return notice


def resume_after_clarification(
    question: str, history: list[tuple[str, str]]
) -> tuple[str, str | None, int | None]:
    """
    Fold a reply to a clarifying question back into the question it answers.

    "UA57" or "Manchester UK" on its own is not a question. If the last
    assistant turn was a clarification, the reply is appended to the
    passenger's original question, so routing and retrieval see the whole
    situation.

    Returns (effective_question, the clarification answered, where the reply starts).
    """
    if len(history) < 2:
        return question, None, None
    (prev_role, prev_user), (last_role, last_text) = history[-2], history[-1]
    if last_role != "assistant" or prev_role != "user" or not is_clarification(last_text):
        return question, None, None
    return f"{prev_user} {question}", last_text, len(prev_user) + 1


# --------------------------------------------------------------------------
# Flight data from another day. AirLabs answers a flight NUMBER with its current
# or most recent operation. A passenger asking about "yesterday's" UA57 must not
# be told today's status or delay as if it were theirs.
# --------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)[a-z]*\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_YESTERDAY_RE = re.compile(r"\b(yesterday|last night)\b", re.I)
_RELATIVE_PAST_RE = re.compile(
    r"\b(last\s+(?:week|weekend|month|year|monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday)|(?:\d+|a|an|one|two|three|four|five|six|a few|few|a couple of|couple of)\s+"
    r"(?:days?|weeks?|months?|years?)\s+ago|earlier this (?:week|month|year))\b", re.I)
_ISO_DATE_RE = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")
_DAY_MONTH_RE = re.compile(rf"\b{_DAY}\s+(?:of\s+)?{_MONTH}(?:\s+(20\d\d))?\b", re.I)
_MONTH_DAY_RE = re.compile(rf"\b{_MONTH}\s+{_DAY}(?:,?\s+(20\d\d))?\b", re.I)


@dataclass(frozen=True)
class DateReference:
    text: str
    day: date | None     # None when the phrase names no single day ("last week")


def referenced_flight_date(question: str, today: date) -> DateReference | None:
    """A past or explicit date the question refers to, or None ("today", no date)."""
    m = _YESTERDAY_RE.search(question)
    if m:
        return DateReference(m.group(0), today - timedelta(days=1))
    m = _RELATIVE_PAST_RE.search(question)
    if m:
        return DateReference(m.group(0), None)
    m = _ISO_DATE_RE.search(question)
    if m:
        try:
            return DateReference(m.group(0), date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            return DateReference(m.group(0), None)
    for rx, day_group, month_group in ((_DAY_MONTH_RE, 1, 2), (_MONTH_DAY_RE, 2, 1)):
        m = rx.search(question)
        if m:
            try:
                month = _MONTHS[m.group(month_group).lower()[:3]]
                year = int(m.group(3)) if m.group(3) else today.year
                d = date(year, month, int(m.group(day_group)))
                if not m.group(3) and d > today:
                    d = date(year - 1, month, d.day)
                return DateReference(m.group(0), d)
            except (ValueError, KeyError):
                return DateReference(m.group(0), None)
    return None


def flight_data_date_note(live: FlightStatus, ref: DateReference) -> str | None:
    """
    None when the record is for the day the passenger means; otherwise the sentence
    that tells them the flight data was used only for the airports.
    """
    record_day = live.flight_date
    if ref.day is not None and record_day == ref.day.isoformat():
        return None
    which = f"is for {record_day}" if record_day else "is for its latest scheduled operation"
    return (f"The flight data I can look up for {live.flight_iata} {which}, which may not be the "
            f"flight you mean (you mentioned \"{ref.text}\"), so I've used it only to identify "
            "the airports, not for its status or delay.")


def found_flight_note(live: FlightStatus, with_status: bool = True) -> str:
    """
    Tell the passenger which flight the lookup found, before anything is built on it:
    "I found DL216: JFK to DSS, cancelled." The passenger can then correct a wrong
    flight number instead of reading an answer about someone else's flight.
    """
    route = " to ".join(x for x in (live.dep_iata, live.arr_iata) if x)
    parts = [p for p in (route, _status_words(live) if with_status else "") if p]
    found = f"I found {live.flight_iata}" + (": " + ", ".join(parts) if parts else "")
    return f"{found}. If that's not your flight, tell me."


def flight_not_found_note(flight_no: str) -> str:
    """
    The lookup ran and AirLabs had no record of the flight. Say so.

    Silence reads as "it never looked": the passenger gave a flight number,
    got an answer opening with a bare assumption, and had no way to tell that
    the route it assumed was not the route their flight actually flew.
    clarification_message() says this on the first round; after it, this note
    is the only place the passenger hears it.
    """
    return (f"I couldn't find flight {flight_no} in the live flight data, so I couldn't "
            "confirm its route or status and the answer below isn't based on them.")


def _status_words(live: FlightStatus) -> str:
    status = (live.status or "").lower()
    delay = live.worst_delay_min
    if status == "cancelled":
        return "cancelled"
    if delay:
        hours, minutes = divmod(int(delay), 60)
        span = f"{hours} h {minutes} min" if hours else f"{minutes} min"
        return f"{status}, delayed about {span}" if status else f"delayed about {span}"
    if not status:
        return ""
    # AirLabs leaves the delay fields empty when it has no delay for the flight;
    # say so, or "scheduled" reads as if the delay were simply unknown.
    return f"{status}, on time" if delay == 0 else f"{status}, no delay reported"


# The passenger's own words say the flight is delayed or cancelled (a claim or a
# question). Same misspellings as jurisdiction._DISRUPTION_RE.
_SAYS_CANCELLED_RE = re.compile(r"\b(cancel\w*|ca(?:nc|n|c)e?l+(?:ed|ing|ation|ations)?)\b", re.I)
_SAYS_DELAYED_RE = re.compile(r"\b(delay\w*|de(?:al|l)y\w*)\b", re.I)


def flight_status_conflict_note(live: FlightStatus, passenger_text: str) -> str | None:
    """
    The sentence that tells the passenger the flight data does not show the delay or
    cancellation they describe ("is my flight delayed?", "my flight was cancelled"),
    or None when there is nothing to contradict. The passenger may have the wrong
    flight or date, or the data may lag the airline, so the sentence invites a
    correction instead of calling the passenger wrong.
    """
    status = (live.status or "").lower()
    if not status or status == "cancelled":
        return None
    if _SAYS_CANCELLED_RE.search(passenger_text):
        claim = "cancelled"
    elif _SAYS_DELAYED_RE.search(passenger_text) and not live.worst_delay_min:
        claim = "delayed"
    else:
        return None
    shown = _status_words(live)
    when = live.dep_estimated or live.dep_time
    if status == "scheduled" and when:
        shown += f" (departure {when} local time" + (f" from {live.dep_iata})" if live.dep_iata else ")")
    airline = AIRLINE_NAMES.get((live.airline_iata or "").upper(), "the airline")
    return (f"From what I can find, {live.flight_iata} isn't {claim}: the flight data shows it "
            f"{shown}. This data can lag behind the airline, so if {airline} has told you it is "
            f"{claim}, tell me what they said.")


def codeshare_note(live: FlightStatus) -> str | None:
    if not live.is_codeshare:
        return None
    seller = AIRLINE_NAMES.get((live.airline_iata or "").upper(), live.airline_iata)
    operator = AIRLINE_NAMES.get(live.operating_airline_iata.upper(), live.operating_airline_iata)
    return (f"{live.flight_iata} is sold by {seller} but operated by {operator}. The airline "
            "responsible for your rights can differ from the one you booked with, so confirm with "
            f"{seller} which rules and policies apply to your ticket.")


# 2 (Session 31): an optional "last_answer" joined the record, so a recognised
# follow-up can be answered from it without generating. Backward compatible - a
# version-1 state simply has no quotable answer and every turn takes the full path.
STATE_VERSION = 2

_log = logging.getLogger("flight_delay.turn")


def _log_turn(answer: Answer) -> None:
    """One log line per answered turn: why it was slow, retried or rejected (Session 17)."""
    d = answer.diagnostics or {}
    # "error" carries the provider's COMPLETE sanitized message. Without it a 429 is
    # just a status and a Retry-After, and those cannot tell tokens-per-minute from
    # tokens-per-day - which is the whole difference between "wait 40 s" and "this
    # account is done for the day" (Session 19: a 2,915 s Retry-After with no way to
    # see which limit produced it).
    calls = [{k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens",
                                     "finish_reason", "attempts", "error_status", "error_code",
                                     "retry_after_s", "error")}
             | {"http_errors": [(e.get("status"), e.get("retry_after_s"))
                                for e in (u.get("http_errors") or []) if isinstance(e, dict)]}
             for u in d.get("usage") or []]
    record = {"outcome": answer.outcome, "intent": answer.intent, "retries": answer.retry_count,
              "timings_ms": {k: round(v) for k, v in (answer.timings_ms or {}).items()},
              "calls": calls, "validation_failures": answer.validation_failures}
    if d.get("uncited_factual_sentences"):
        record["uncited"] = d["uncited_factual_sentences"]
    # Which template fired and whether it had anything to quote: the only way to
    # see, from the logs, that a turn was answered without a model call and
    # whether the answer it was built from actually covered what was asked.
    if d.get("followup"):
        record["followup"] = {k: d["followup"][k] for k in ("template", "had_material")}
    # The draft the validator threw away. It is deliberately NOT sent to the passenger
    # and NOT stored in `messages` (that table is the conversation, and this text was
    # never part of it), so without this line it exists nowhere: the only trace of a
    # rejected answer was the uncited sentence list. That makes every rejection cost a
    # full generation (~6,700 tokens on the free tier) and teach almost nothing about
    # what the model actually wrote (Session 19).
    if d.get("rejected_answer"):
        record["rejected_answer"] = d["rejected_answer"]
    _log.info("turn %s", json.dumps(record, default=str))


def turn_state(plan: TurnPlan, last_answer: dict | None = None) -> dict:
    """
    What a conversation has settled, for the next turn (JSON-serialisable).

    `last_answer` is the grounded answer a follow-up may quote from
    (answer_record). It is carried FORWARD unchanged across follow-up turns, so a
    run of them all quote the one answer that was actually retrieved and
    validated, never each other.
    """
    state = {
        "version": STATE_VERSION,
        "airline": plan.flt.get("airline_scope"),
        "unsupported_airline": plan.unsupported_airline,
        "flight_no": plan.flight_no,
        "route": asdict(plan.route),
        "flt": {k: v for k, v in plan.flt.items()
                if k in ("airline_scope", "jurisdiction_scope", "jurisdiction_governing")},
    }
    if last_answer:
        state["last_answer"] = last_answer
    return state


def answer_record(answer: Answer) -> dict | None:
    """
    The part of a finished answer a later follow-up can be built from, or None
    when this turn produced nothing quotable.

    Only a SHOWN, validated answer qualifies: a gated refusal, a validation
    failure and an LLM error all state nothing a passenger may be told twice. The
    text stored is the model's own (diagnostics["model_answer"]), not the shown
    text, because the code openers prepended by with_openers - the route
    assumption, the flight-data note - are this turn's framing and would read as
    stale quotes on the next one.
    """
    if answer.outcome not in ("answered", "abstained"):
        return None
    text = (answer.diagnostics or {}).get("model_answer") or answer.text
    if not text or len(text) > followups.STORED_ANSWER_MAX_CHARS:
        return None
    return {"text": text,
            "citations": [{k: getattr(c, k) for k in _CITATION_FIELDS} for c in answer.citations],
            # What the answer was ABOUT, so a follow-up naming another disruption is not
            # answered from it (Session 37: "How much compensation for denied boarding?"
            # after a cancellation answer re-quoted the cancellation notice rule).
            "disruptions": disruptions_named(answer.question, answer.live_data)}


_CITATION_FIELDS = ("marker", "chunk_id", "doc_title", "section_id", "source_url")


def followup_citations(record: dict, markers: list[str]) -> list[Citation]:
    """The stored citations for the markers a follow-up actually quoted, in that order."""
    by_marker = {c.get("marker"): c for c in record.get("citations") or [] if isinstance(c, dict)}
    return [Citation(**{k: str(by_marker[m].get(k) or "") for k in _CITATION_FIELDS})
            for m in markers if m in by_marker]


def _route_from_state(d: dict) -> RouteJurisdiction:
    def tup(v):
        return tuple(tup(x) for x in v) if isinstance(v, list | tuple) else v

    return RouteJurisdiction(**{k: tup(v) for k, v in d.items()})


def _names_route(route: RouteJurisdiction) -> bool:
    """Does this message itself say anything about where the flight went?"""
    return bool(route.origin_region or route.destination_region or route.origin_place
                or route.destination_place or route.endpoints or route.ambiguous_places
                or route.unknown_places or route.domestic)


@dataclass
class TurnPlan:
    """What to do with one passenger message, decided before any retrieval."""

    question: str                 # effective question (original + clarification reply)
    intent: Intent
    flight_no: str | None
    live: FlightStatus | None
    route: RouteJurisdiction      # the assumed route, when an assumption was made
    flt: dict
    clarification: str | None     # ask this instead of answering
    assumption: str | None        # open the answer with this sentence
    unsupported_airline: str | None = None  # IATA code of a carrier this assistant does not cover
    unsupported_reply: str | None = None    # say this instead of answering (unsupported carrier)
    notice: str | None = None               # open the answer with this, before any assumption
    # Sentences about the flight data itself (another day's operation, a codeshare),
    # shown with the answer and given to the model as a FLIGHT DATA NOTE.
    flight_note: str | None = None
    # False when the record is for a different day than the passenger means: its
    # airports still route the question, its status and delay are not shown.
    live_status_usable: bool = True
    inherited: bool = False                 # route/carrier carried over from earlier turns
    # The passenger said (or asked whether) the flight is delayed or cancelled and the
    # same-day flight data shows otherwise; flight_note carries the sentence saying so.
    status_conflict: bool = False


def plan_turn(
    question: str,
    history: list[tuple[str, str]],
    fetch_flight: Callable[[str], FlightStatus | None],
    flt: dict | None = None,
    prior: dict | None = None,
    today: date | None = None,
) -> TurnPlan:
    """
    Route one message: resume a clarification, look up the flight, detect the
    route, then ask, answer, or answer on a stated assumption.

    `prior` is the state an earlier answered turn in this conversation settled
    (turn_state). A follow-up that names no carrier, flight or place ("what about
    a hotel?") keeps that route and carrier instead of losing its filters or being
    asked again; one that names something new is routed afresh, keeping only the
    carrier when it names none.

    A pure function of its inputs (plus the flight lookup), so the golden-set
    builder and the eval harnesses run exactly what production runs.
    """
    today = today or datetime.now(UTC).date()
    question, answered, reply_start = resume_after_clarification(question, history)
    intent, flight_no = classify(question)
    carrier = resolve_airline(question, flight_no)
    if prior and answered is None and carrier is None and flight_no is None:
        carrier = prior.get("airline")
        if carrier and (flt is None or "airline_scope" not in flt):
            flt = {**(flt or {}), "airline_scope": carrier}
    # A follow-up ("do you know my departure airport?", "what about a hotel?") that
    # names no flight, carrier or place is about the flight already looked up in this
    # conversation: keep it, or the model answers without the flight data (Session 17).
    carried_flight = False
    if (flight_no is None and prior and answered is None and prior.get("flight_no")
            and carrier in (None, prior.get("airline"))
            and not _names_route(detect_jurisdictions(question))):
        flight_no, carried_flight = prior["flight_no"], True
        carrier = carrier or prior.get("airline")
    unsupported = carrier if carrier is not None and carrier not in SUPPORTED_AIRLINES else None
    # Only a supported carrier's flight number is looked up (or quoted back as
    # "couldn't find flight X"): BA117 never costs an AirLabs call.
    lookup_no = flight_no if airline_of_flight(flight_no) in SUPPORTED_AIRLINES else None
    live = fetch_flight(lookup_no) if lookup_no else None

    notes: list[str] = []
    live_status_usable = True
    status_conflict = False
    if live is not None:
        ref = referenced_flight_date(question, today)
        date_note = flight_data_date_note(live, ref) if ref else None
        if date_note:
            live_status_usable = False
        if not carried_flight:
            # A record for another day still gives the airports, never its status.
            notes.append(found_flight_note(live, with_status=live_status_usable))
            if live_status_usable:
                # "Delayed" may have been said a turn or two before the flight number.
                said = " ".join([question, *(text for role, text in history[-4:] if role == "user")])
                if conflict := flight_status_conflict_note(live, said):
                    notes.append(conflict)
                    status_conflict = True
        if date_note:
            notes.append(date_note)
        if cs := codeshare_note(live):
            notes.append(cs)
    elif lookup_no and not carried_flight:
        # Looked up and missed (no record, quota guard or open circuit). The
        # answer is about to be built on an assumed route; say where that came
        # from. See flight_not_found_note().
        notes.append(flight_not_found_note(lookup_no))

    caller_scoped = flt is not None and "jurisdiction_scope" in flt
    flt, route = route_filters(question, flight_no, live, flt, reply_start=reply_start)

    # A record that does not say where the flight departed (or, leaving Europe,
    # where it was going) cannot decide the law; it must not skip the question.
    # An airport code the table does not list counts as not known (region_of -> None).
    dep_region = region_of(live.dep_iata) if live is not None else None
    live_route_known = dep_region is not None and (
        region_of(live.arr_iata) is not None or dep_region == "US")

    inherited = False
    if (prior and answered is None and not caller_scoped and flight_no is None
            and not _names_route(route) and prior.get("route")):
        route = _route_from_state(prior["route"])
        flt = {**{k: v for k, v in flt.items()
                  if k not in ("jurisdiction_scope", "jurisdiction_governing")},
               "jurisdiction_scope": list(route.scope),
               "jurisdiction_governing": list(route.governing)}
        inherited = True

    clarification = assumption = unsupported_reply = notice = None
    shown_no = flight_no if unsupported and airline_of_flight(flight_no) == unsupported else None
    if inherited:
        pass
    elif unsupported and (route.needs_clarification or (intent == "status" and not route.direction_known)):
        # Nothing to look up, no airline policy to give, and the route that would
        # decide the law is unknown: say so, rather than run a clarification
        # round whose answer could only be government rules anyway.
        unsupported_reply = unsupported_airline_message(unsupported, shown_no)
    elif not caller_scoped and not live_route_known:
        if route.needs_clarification and answered is None:
            clarification = clarification_message(route, lookup_no, found_without_route=live is not None)
        elif route.needs_clarification or answered is not None:
            # Asked once already. Answer, and say what the answer assumes:
            # either a default fills the gap, or the route came from the
            # passenger's words rather than flight data.
            route, assumption = assume_route(route)
            flt = _with_route({k: v for k, v in flt.items()
                               if k not in ("jurisdiction_scope", "jurisdiction_governing")}, route)

    if unsupported and unsupported_reply is None:
        notice = unsupported_airline_notice(unsupported, shown_no, route)
        if intent == "status":
            # There is no flight status to give; the rules for the stated route
            # are the only answer available, so retrieve them.
            intent = "policy"
    if intent == "status" and not live_status_usable:
        # The record is for another day: its status is not an answer.
        intent = "policy"

    return TurnPlan(question, intent, flight_no, live, route, flt, clarification, assumption,
                    unsupported, unsupported_reply, notice,
                    flight_note=" ".join(notes) or None, live_status_usable=live_status_usable,
                    inherited=inherited, status_conflict=status_conflict)


def classify(question: str) -> tuple[Intent, str | None]:
    """
    Decide which evidence the question needs.

    Returns the intent and the extracted flight number, if any.

    The presence of a flight designator is the strongest single signal: it means
    the user is asking about a SPECIFIC flight, which no amount of policy text
    can answer alone.
    """
    flight = extract_flight_number(question)
    wants_policy = bool(_POLICY_HINT.search(question))
    wants_status = bool(_STATUS_HINT.search(question))

    if flight and wants_policy:
        return "hybrid", flight
    if flight and wants_status:
        return "status", flight
    if flight:
        # A bare flight number with no other signal - most likely a status check.
        return "status", flight
    return "policy", None


# --------------------------------------------------------------------------
# Retrieval query (Session 22). Routing already knows facts the passenger's
# sentence does not carry ("UA 9901 is showing a big delay" says nothing of
# Heathrow, the UK or 290 minutes late at arrival). The SEARCH gets them as a
# short preface; the model, the gate and the log keep the passenger's words.
# Every fact comes from the route resolver, the flight record or the question
# itself: nothing is inferred about the law (no distance band, no amount).
# --------------------------------------------------------------------------

_REGION_WORDS = {"US": "United States", "EU": "European Union", "UK": "United Kingdom",
                 "OTHER": "outside the US, EU and UK"}

# Disruptions, in the order they are named. Denied boarding is also what lets
# denied-boarding text into the pool (retrieval.is_denied_boarding_chunk).
_DISRUPTIONS = (
    ("denied boarding", _term_matcher(
        r"bump(?:ed|ing)?", r"denied (?:me )?boarding", r"deny(?:ing)? (?:me )?boarding",
        r"denied me", r"over-?sold", r"over-?book(?:ed|ing)?", r"oversales?",
        r"volunteer(?:s|ed|ing)?", r"involuntar\w*", r"gave (?:away )?my seat")),
    ("cancellation", _term_matcher(
        r"cancel(?:s|ed|led|ing|ling|lation|lations)?", r"ca(?:nc|n|c)e?l+(?:ed|ing|ation|ations)?")),
    ("delay", _term_matcher(r"delay(?:s|ed|ing)?", r"late", r"hours? late", r"waiting")),
    ("missed connection", _term_matcher(r"missed (?:(?:my|our|the|a) )?connect\w*", r"misconnect\w*")),
    ("diversion", _term_matcher(r"divert(?:s|ed|ing)?", r"diversion")),
    # Session 23: "sat on the ground ... after landing and we couldn't get off" (dom-34)
    # named no tarmac delay, so the search never looked for 14 CFR 259.4.
    ("tarmac delay", _term_matcher(
        r"tarmac", r"stuck on the (?:plane|aircraft)", r"(?:sat|sitting|stuck|held) on the ground",
        r"could(?:n't|n’t| not) (?:get off|deplane)", r"unable to (?:get off|deplane)",
        r"(?:not|never) (?:allowed|let) (?:to )?(?:get )?off")),
    ("baggage", _term_matcher(r"bags?", r"baggage", r"luggage", r"suitcases?")),
    ("downgrade", _term_matcher(r"downgrad\w*")),
)
# An open "what am I owed?" names no remedy; these are the rights to look for.
_OPEN_ENTITLEMENT = _term_matcher(
    r"owe[ds]?", r"entitle(?:d|ment|ments)?", r"rights?", r"options?", r"do (?:i|we) get",
    r"get anything", r"what (?:do|can) (?:i|we) do", r"compensat\w*", r"what does \w+ (?:provide|give)",
    r"what does (?:\w+ )+give (?:me|us)", r"required to (?:provide|give|do)")
# A question about the complaint process, whatever disruption it mentions: "Delta never
# responded to my written complaint about a cancelled flight. How long do they have?"
# (dom-44) is not a cancellation question, and searching it as one found Delta's
# cancellation pages instead of 14 CFR 259.7 (Session 23).
_COMPLAINT = _term_matcher(
    r"complaints?", r"complain(?:ed|ing)?", r"how long do (?:they|\w+) have",
    r"(?:never|not|hasn't|haven't|didn't|did not|has not|have not) (?:yet )?(?:responded|replied|answered)")
_REMEDIES = {
    "delay": "compensation for a long delay at arrival, meals and care while waiting, refund",
    "cancellation": "cancellation compensation and notice, refund or re-routing, care while waiting",
    "denied boarding": "denied boarding compensation, re-routing or refund, care",
    "missed connection": "delay at final destination, re-routing, refund",
    "diversion": "hotel, meals and transport after a diversion",
    "tarmac delay": "food and water, lavatories, medical attention, status updates, deplaning",
}

# Remedy lanes (Session 23). One search for "what am I owed?" finds the article
# with the most delay wording (UK261 Art 6) and misses the premises beside it
# (Art 7's amounts, the care and refund articles): live-01. Each lane is a small
# extra search over ONE governing regime's government texts, one per remedy; its
# ranking joins the final rank fusion as one more vote (retrieval.HybridRetriever,
# lane_candidates_k). Lanes name remedies, never article numbers or amounts.
_EUROPE_LANES = {
    "delay": ("compensation amount by flight distance for a long delay at arrival",
              "right to care: meals, refreshments and accommodation during a delay",
              "reimbursement or re-routing when a flight is delayed five hours or more"),
    # The CONDITIONS (notice, re-routing times, the exception) are the trigger lane's
    # since Session 37; aimed at them too, this lane seated guidelines 4.4.4 beside
    # Article 5 and Article 7's amounts lost their slot. Probed: EU Article 7 first,
    # UK the CAA's cancellation-notice compensation table first.
    "cancellation": ("compensation for a cancelled flight: the amount by flight distance",
                     "reimbursement or re-routing after a cancellation",
                     "right to care after a cancellation"),
    "denied boarding": ("denied boarding compensation amount",
                        "reimbursement or re-routing after denied boarding"),
    "missed connection": ("compensation for arriving late at the final destination on connecting flights",),
}
_US_LANES = {
    "delay": ("refund when a flight is significantly delayed or changed",),
    "cancellation": ("refund when the airline cancels a flight",),
    "denied boarding": ("involuntary denied boarding compensation amount",),
    "tarmac delay": ("tarmac delay: food, water, lavatories, medical attention and deplaning",),
}
_COMPLAINT_LANE = "deadline for an airline to acknowledge and respond to a written consumer complaint"

# Scope lanes (Session 25). The remedy lanes search the GOVERNING regimes, which is
# right for remedies and wrong for scope: "EU261 does not apply to this flight" is a
# claim under system_prompt.md Rule 3, validate_answer rejects it uncited, and on a
# US->EU flight the regime whose scope has to be cited is precisely the one that does
# not govern. So the scope lane runs over route.SCOPE. Each names the legal concept,
# never "Article 3": the regulators' own "does this law apply to my flight" pages
# answer it as well as the article does (required.py counts them as equivalent).
_SCOPE_QUERIES = {
    # Aimed at Part 260's "covered flight" definition, not at Part 259's applicability:
    # measured on the index, "which rules cover which flights" returns 259.2 and 250.2
    # first and 260.2 seventh, and 260.2 is the section that says which itineraries the
    # refund rules reach (other-01, intra-07, us-eur-01).
    "US": "definition of a covered flight for the US refund rules: which itineraries, "
          "carriers and tickets are covered, to, from or within the United States",
    "UK": "when these passenger rights apply to a flight: departing from or arriving in "
          "the United Kingdom, and whether the operating carrier is a UK or EU carrier",
    "EU": "geographical scope: when these passenger rights apply to a flight, departing "
          "from or arriving in the European Union, and whether the operating carrier is "
          "a Community carrier",
}

# US procedural lanes (Session 25): 14 CFR sections that answer a whole question on
# their own and that no remedy lane can reach, because the question's wording is not
# remedy wording ("do I have to accept it?" is not "refund"). Each fires only on its
# own trigger, so a question that is not about the procedure pays nothing for it.
_PROCEDURAL_LANES: tuple[tuple[re.Pattern[str], str], ...] = (
    # dom-17, dom-21: 14 CFR 260.7, the passenger must affirmatively accept a credit.
    (_term_matcher(r"e-?credits?", r"credits?", r"vouchers?", r"have to accept",
                   r"must (?:i|we) accept", r"do (?:i|we) have to take",
                   r"cash (?:back|instead)", r"money back instead"),
     "a consumer must affirmatively accept a travel credit or voucher offered "
     "instead of a refund"),
    # dom-24: 14 CFR 260.2, what "prompt" means and in which form the money comes back.
    (_term_matcher(r"when (?:will|do|can) (?:i|we) get", r"how long (?:does|will|do|until)",
                   r"paid cash", r"how soon", r"still waiting"),
     # The DEFINITION, not the duty: "deadline for a prompt refund" returns 260.10
     # first and 260.2 fifth, and 260.2 is where "prompt refund means" lives.
     "definition of a prompt refund: how many business days after it becomes due, and "
     "in which form of payment it must be made"),
    # dom-25: 14 CFR 250.8, when and in what form denied boarding compensation is paid.
    (_term_matcher(r"bumped", r"over-?sold", r"over-?sales?", r"denied boarding",
                   r"involuntarily denied"),
     "when denied boarding compensation must be paid, and whether by cash or cheque "
     "on the day of the flight"),
    # dom-29: 14 CFR 250.6, the exceptions that remove the entitlement entirely.
    (_term_matcher(r"smaller (?:plane|aircraft|equipment)", r"equipment (?:swap|change|substitution)",
                   r"swapped", r"substituted", r"operational reasons?", r"weight and balance"),
     "exceptions: when denied boarding compensation is not payable, including "
     "substitution of smaller equipment for operational or safety reasons"),
    # law-02: 14 CFR 259.5, the carrier must adhere to its own customer service plan.
    (_term_matcher(r"conditions of carriage", r"contract of carriage", r"customer service plan",
                   r"sole obligation", r"(?:their|the) contract (?:says|states)",
                   r"polic(?:y|ies) says?"),
     "an air carrier must adopt and adhere to the commitments in its customer "
     "service plan"),
)

# The carriers a passenger names in a question, for _AIRLINE_EVIDENCE below: every
# name the tool knows, its first word ("American", "Delta"), and the generic words.
_CARRIER_WORDS = "|".join(sorted(
    {re.escape(w) for name in AIRLINE_NAMES.values() for w in (name, name.split()[0])}
    # The code too: "What does AA owe me?" (dom-15) names the carrier as well as
    # "What does American provide?" (dom-37) does.
    | {re.escape(code) for code in AIRLINE_NAMES}
    | {"the airline", "the carrier", "they", "it"}, key=len, reverse=True))

# Questions the carrier's own pages cannot answer (Session 25). The airline quota
# exists so an answer says what the carrier PROMISED as well as what the law requires;
# on "how much does UK261 owe me?" the second reserved airline slot buys a contract
# page the answer will not cite, and costs a legal premise. Naming the carrier is not
# asking about its policy ("my Delta flight was delayed"), so carrier names are absent.
_AIRLINE_EVIDENCE = _term_matcher(
    r"hotels?", r"meals?", r"foods?", r"drinks?", r"refreshments?", r"vouchers?",
    r"accommodat(?:e|ed|ion|ions)", r"lodging", r"re-?book(?:s|ed|ing)?", r"re-?protect\w*",
    r"amenit(?:y|ies)", r"polic(?:y|ies)", r"conditions of carriage", r"contract of carriage",
    r"customer service plan", r"commitments?", r"promis(?:e|es|ed)", r"e-?credits?",
    r"credits?", r"miles", r"points", r"upgrades?", r"seats?", r"baggage", r"bags?",
    r"luggage", r"taxis?", r"ground transportation", r"transport(?:ation)?",
    # "what does American provide", "does Delta owe me a hotel": the SUBJECT has to be
    # a carrier. With a bare \w+ this matched "does UK261 owe me", which is the
    # opposite kind of question - the one the airline's own pages cannot answer.
    *(rf"(?:what |how )?(?:does|do|will|can|must) (?:{_CARRIER_WORDS}) "
      r"(?:provide|give|offer|owe|cover|put|pay|do|have to)",),
)


def disruptions_named(question: str, live: FlightStatus | None = None) -> list[str]:
    """The disruptions the question (or a usable flight record) names, in table order."""
    found = [name for name, rx in _DISRUPTIONS if rx.search(question)]
    if live is not None:
        cancelled = (live.status or "").lower() == "cancelled"
        if cancelled and "cancellation" not in found:
            found.append("cancellation")
        # A cancelled record's delay minutes describe no delay the passenger lived
        # through (live UA169, Session 37: "cancelled, delay 235 min"). Counted, they
        # added four delay lanes that crowded the cancellation ones out of the seats.
        if (not cancelled and max(live.dep_delayed or 0, live.arr_delayed or 0) >= 15
                and "delay" not in found):
            found.append("delay")
    return found


def excluded_topics(question: str, live: FlightStatus | None = None) -> tuple[str, ...]:
    """Topics whose sections the search leaves out (retrieval.TOPIC_TESTS)."""
    kinds = disruptions_named(question, live)
    out = []
    if "denied boarding" not in kinds:
        out.append("denied_boarding")
    if "delay" in kinds and "cancellation" not in kinds:
        out.append("cancellation")
    return tuple(out)


def remedy_lanes(question: str, plan: TurnPlan,
                 live: FlightStatus | None) -> tuple[tuple[str, str], ...]:
    """(jurisdiction, query) per remedy lane for this question; () when none apply."""
    governing = tuple(plan.route.governing or ())
    if _COMPLAINT.search(question):
        return (("US", _COMPLAINT_LANE),) if "US" in governing else ()
    kinds = disruptions_named(question, live)
    if not (_OPEN_ENTITLEMENT.search(question) or "tarmac delay" in kinds):
        return ()
    lanes: list[tuple[str, str]] = []
    for juris in governing:
        table = _US_LANES if juris == "US" else _EUROPE_LANES
        name = REGIME_NAMES.get(juris, juris)
        for kind in kinds:
            lanes += [(juris, f"{name} {q}") for q in table.get(kind, ())]
    return tuple(dict.fromkeys(lanes))


def scope_lanes(plan: TurnPlan) -> tuple[tuple[str, str, str], ...]:
    """(jurisdiction, query) per regime the route TOUCHES; () on a US-only route."""
    scope = tuple(plan.route.scope or ())
    # Nobody asks whether US law covers a US domestic flight, and Part 260's coverage
    # section would take the slot from the rule that was actually asked about. The
    # question is live the moment the itinerary crosses a border (other-01, intra-07)
    # - and when US law is only POSSIBLE on a route leaving from outside the US, EU
    # and UK (Session 37, live "my Toronto flight": with no coverage text the model
    # stated US refund law applies outright).
    if not scope or (set(scope) == {"US"} and route_law_note(plan.route) is None):
        return ()
    return tuple((j, f"{REGIME_NAMES.get(j, j)}: {_SCOPE_QUERIES[j]}", "scope")
                 for j in scope if j in _SCOPE_QUERIES)


# Trigger lanes (Session 37): the article that makes a disruption compensable at all.
# For an EU/UK cancellation that is Article 5 - compensation, care AND re-routing, and
# the advance-notice exceptions - which no remedy lane seats: the compensation lane
# ranks Article 7 (the amounts) first and Article 5 third, and one lane makes one
# offer. Without it the live UA169 VCE->EWR answer cited the extraordinary-
# circumstances exception to Article 7 and offered the cash "if you prefer" it to a
# refund. Aimed by probing the index: this wording returns Article 5 FIRST for both
# EU261 and UK261 (0.814 / 0.815), with the Commission's 3.2.6 "Cancellation of a
# flight gives: (i)... (ii)... (iii)..." second for EU.
_TRIGGER_QUERIES = {
    "cancellation": ("cancellation of a flight: the rights the passengers concerned shall be "
                     "offered - assistance, care and compensation - unless informed of the "
                     "cancellation in advance"),
}


def trigger_lanes(question: str, plan: TurnPlan,
                  live: FlightStatus | None) -> tuple[tuple[str, str, str], ...]:
    """One seat-taking lane per governing EU/UK regime and disruption with a trigger article."""
    if _COMPLAINT.search(question) or not _OPEN_ENTITLEMENT.search(question):
        return ()
    kinds = disruptions_named(question, live)
    return tuple((j, f"{REGIME_NAMES.get(j, j)} {_TRIGGER_QUERIES[k]}", "trigger")
                 for j in (plan.route.governing or ()) if j in ("EU", "UK")
                 for k in kinds if k in _TRIGGER_QUERIES)


def route_law_note(route: RouteJurisdiction) -> str | None:
    """
    What the model must be told when the route names a place outside the US, EU and
    UK and no regime governs. Routing already knows this; without the note the model
    saw only US sources and applied them as settled (Session 37, "My Toronto flight
    was cancelled", flight number not found). It states only what routing decided -
    which regimes are established, which are merely possible - never a rule's content.

    A flight DEPARTING outside the US, EU and UK is never settled under US DOT, even
    into the US (user's decision, 2026-09-22): the answer says US rules MAY apply and
    sends the passenger to the departure country's government site and the airline.
    """
    if route.governing:
        return None
    refer = ("Tell the passenger to check the passenger-rights website of the departure "
             "country's government, and to ask the airline. That country's own rules are not "
             "in the sources; say so, and do not state them.")
    if route.origin_region == "OTHER":
        if "US" in (route.scope or ()):
            return ("LAW NOTE: This flight departs from outside the US, EU and UK, so no law in "
                    "the sources is established for it and you cannot be sure which rules apply. "
                    "US rules MAY apply: say that they may, never that they do, and cite the text "
                    "that defines which flights the US rules cover. " + refer)
        return ("LAW NOTE: This flight departs from outside the US, EU and UK, and none of the "
                "US, EU or UK rules in the sources governs it. " + refer)
    if route.destination_region == "OTHER" and "US" in (route.scope or ()):
        return ("LAW NOTE: No law in the sources is established as governing this flight. It "
                "involves a place outside the US, EU and UK, and the route is not fully known. "
                "US rules apply only if the flight departs from or arrives in the United States: "
                "answer conditionally (\"IF your flight was to or from the US, ...\") and cite the "
                "text that defines which flights the US rules cover. The rules of other countries "
                "are not in the sources; say so, and do not state them.")
    return None


def procedural_lanes(question: str) -> tuple[tuple[str, str, str], ...]:
    """US procedural lanes whose own wording this question uses; () when none."""
    return tuple(("US", query, "procedural") for rx, query in _PROCEDURAL_LANES
                 if rx.search(question))


def search_lanes(question: str, plan: TurnPlan, live: FlightStatus | None,
                 settings) -> tuple[tuple[str, str, str], ...]:
    """
    Every extra dense search this turn runs, as (jurisdiction, query, kind).

    The kind is what retrieval.HybridRetriever reserves a seat by: a remedy lane
    votes in the fusion and nothing more, while a scope or procedural lane's best
    hit IS the premise the answer is missing (measured on the index: the UK scope
    lane returns Article 3 first, the acceptance lane returns 260.7 first), so a
    vote it loses to two better-fused chunks costs the whole case.
    """
    if not getattr(settings, "lane_candidates_k", 0):
        return ()
    # Scope first, then the procedural lanes, then the trigger lanes (Session 37), then
    # the remedy lanes: the order is the order seats are handed out in, and a seat
    # that does not fit is simply not taken.
    # A missing scope premise blocks the answer under Rule 3; a missing remedy article
    # only makes it thinner.
    lanes: list[tuple[str, str, str]] = []
    if getattr(settings, "scope_lanes", False):
        lanes += list(scope_lanes(plan))
    if getattr(settings, "procedural_lanes", False) and "US" in (plan.route.scope or ()):
        lanes += list(procedural_lanes(question))
    if getattr(settings, "trigger_lanes", False):
        lanes += list(trigger_lanes(question, plan, live))
    lanes += [(j, q, "remedy") for j, q in remedy_lanes(question, plan, live)]
    seen: dict[tuple[str, str], tuple[str, str, str]] = {}
    for lane in lanes:
        seen.setdefault(lane[:2], lane)
    return tuple(seen.values())


def airline_quota(question: str, settings) -> int:
    """Airline slots to reserve: the full quota, or the smaller legal/procedural one."""
    full = settings.min_airline_sources
    legal = getattr(settings, "min_airline_sources_legal", None)
    if legal is None or _AIRLINE_EVIDENCE.search(question):
        return full
    return min(full, legal)


def _minutes(m: int) -> str:
    h, mm = divmod(int(m), 60)
    return f"{h} h {mm} min" if h and mm else f"{h} h" if h else f"{mm} min"


def retrieval_query(question: str, plan: TurnPlan, live: FlightStatus | None) -> str:
    """The text the retriever searches with: known facts, then the question itself."""
    facts: list[str] = []
    # The record's airports route the question even when its status is for another
    # day (plan.live); its delays are used only when usable (`live`).
    rec = plan.live
    carrier = (plan.flt or {}).get("airline_scope") or (rec.airline_iata if rec else None)
    name = AIRLINE_NAMES.get((carrier or "").upper())
    route = plan.route

    def end(code: str | None, place, region: str | None) -> str | None:
        where = code or (place[0] if place else None)
        region = region or (place[1] if place else None)
        word = _REGION_WORDS.get(region or "")
        if where and word:
            return f"{where} ({word})"
        return where or word

    dep = end(rec.dep_iata if rec else None, route.origin_place,
              region_of(rec.dep_iata) if rec else route.origin_region)
    arr = end(rec.arr_iata if rec else None, route.destination_place,
              region_of(rec.arr_iata) if rec else route.destination_region)
    flight = " ".join(x for x in (name, "flight") if x)
    if dep == arr == _REGION_WORDS["US"] or ((dep is None or arr is None) and route.domestic):
        # Both ends known only as "somewhere in the US".
        dep = arr = None
        flight = f"{name or 'US'} domestic flight within the United States"
    elif dep or arr:
        flight += (f" from {dep}" if dep else "") + (f" to {arr}" if arr else "")
    if flight != "flight":
        facts.append(flight)

    complaint = bool(_COMPLAINT.search(question))
    kinds = [] if complaint else disruptions_named(question, live)
    if complaint:
        facts.append("issue: consumer complaint to the airline; rights: deadlines to acknowledge "
                     "and to answer a written complaint")
    if kinds:
        facts.append("disruption: " + ", ".join(kinds))
    if live is not None:
        if live.dep_delayed:
            facts.append(f"departure delay {_minutes(live.dep_delayed)}")
        if live.arr_delayed:
            facts.append(f"arrival delay {_minutes(live.arr_delayed)}")
    if route.governing:
        facts.append("law: " + ", ".join(REGIME_NAMES[g] for g in route.governing if g in REGIME_NAMES))
    if _OPEN_ENTITLEMENT.search(question):
        wants = [_REMEDIES[k] for k in kinds if k in _REMEDIES]
        if wants:
            facts.append("rights: " + "; ".join(dict.fromkeys(wants)))
    if not facts:
        return question
    return f"{question}\n{'. '.join(facts)}."


class RagPipeline:
    def __init__(self, retriever, llm, tool, settings, store=None, gate=None):
        self.retriever = retriever
        self.llm = llm
        self.tool = tool
        self.s = settings
        self.store = store
        # Confidence gate runs BEFORE generation. Adapted from the corrective-RAG
        # relevance-grader pattern, but without an LLM call: it reuses the
        # cross-encoder score we already paid for. See confidence.py.
        self.gate = gate or ConfidenceGate(
            min_rerank_score=settings.min_rerank_score,
            min_score_margin=settings.min_score_margin,
            min_grounding_ratio=settings.min_grounding_ratio,
            enabled=settings.confidence_gate_enabled,
        )

    # ------------------------------------------------------------ evidence
    def _fetch_flight(self, flight_no: str):
        if not flight_no:
            return None
        try:
            fs = self.tool.get_flight(flight_no)
        except QuotaExceeded:
            airlabs_calls_total.labels(source="blocked").inc()
            return None
        airlabs_calls_total.labels(source="miss" if fs is None else fs.source).inc()
        with contextlib.suppress(Exception):
            airlabs_quota_remaining.set(self.tool.quota.remaining())
        return fs

    # ----------------------------------------------------------------- run
    def run(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        flt: dict | None = None,
        allow_retry: bool = True,
        allow_followup: bool = True,
    ) -> Answer:
        """
        One passenger message, end to end.

        `allow_followup=False` forces the full retrieve-and-generate path even for
        a message followups.match() recognises. The golden replay harness passes
        it (evals/golden/replay.py) so that no eval case can be scored on a
        template reply: retrieval, generation and the golden set must keep
        measuring the pipeline they were built to measure.
        """
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        # The message as the passenger typed it. `question` is rebound below to
        # plan.question, which may have an earlier turn joined to it; follow-up
        # matching wants the words of THIS message only.
        raw_question = question

        # Loaded first: a reply to a clarifying question ("UA57") only means
        # something joined to the question it answers, and a follow-up keeps the
        # route the conversation already settled.
        history: list[tuple[str, str]] = []
        prior = None
        if conversation_id and self.store is not None:
            try:
                history = self.store.history(conversation_id)
            except Exception:
                history = []
            try:
                prior = self.store.load_state(conversation_id) if hasattr(self.store, "load_state") else None
            except Exception:
                prior = None

        def timed_fetch(flight_no):
            t0 = time.perf_counter()
            fs = self._fetch_flight(flight_no)
            timings["flight_lookup"] = (time.perf_counter() - t0) * 1000
            rag_stage_duration_seconds.labels(stage="flight_lookup").observe(
                timings["flight_lookup"] / 1000
            )
            return fs

        plan = plan_turn(question, history, timed_fetch, flt, prior=prior)
        question, intent, flt = plan.question, plan.intent, plan.flt
        live = plan.live if plan.live_status_usable else None

        def finish(answer: Answer, *, save_state: bool, last_answer: dict | None = None) -> Answer:
            elapsed = time.perf_counter() - t_start
            rag_end_to_end_seconds.observe(elapsed)
            timings["total"] = elapsed * 1000
            answer.timings_ms = timings
            answer.prompt_sha256 = SYSTEM_PROMPT_SHA256
            rag_requests_total.labels(intent=answer.intent, outcome=answer.outcome).inc()
            self._save_turn(conversation_id, question, answer.text,
                            [c.__dict__ for c in answer.citations])
            if save_state:
                # A turn that produced nothing quotable (gated, rejected, failed)
                # keeps the last answer that DID, rather than erasing the
                # conversation's ability to answer a follow-up.
                self._save_state(conversation_id, plan,
                                 last_answer or answer_record(answer)
                                 or (prior or {}).get("last_answer"))
            _log_turn(answer)
            return answer

        # -- clarify -----------------------------------------------------
        # The route is unknown and it decides whether €600 is owed. Ask for
        # what is missing (flight number, airports, which Manchester, which
        # country) instead of guessing.
        if plan.clarification is not None:
            return finish(Answer(question=question, intent="clarify", text=plan.clarification,
                                 live_data=live, clarification=True, outcome="clarified"),
                          save_state=False)

        # -- unsupported airline ------------------------------------------
        # Not one of the four carriers, and nothing carrier-independent can be
        # said yet (a status check, or a route that is still unknown).
        if plan.unsupported_reply is not None:
            return finish(Answer(question=question, intent="unsupported",
                                 text=plan.unsupported_reply,
                                 unsupported_airline=plan.unsupported_airline,
                                 outcome="declined_unsupported"),
                          save_state=False)

        openers = [s for s in (plan.notice, plan.flight_note, plan.assumption) if s]

        def with_openers(text: str) -> str:
            # Prepended by code, not left to the model: a small model will not
            # reliably open with them, and the passenger must see them.
            for opening in reversed(openers):
                if not text.lstrip().startswith(opening[:30]):
                    text = f"{opening}\n\n{text}"
            return text

        # -- follow-up ---------------------------------------------------
        # A short, recognised move after a grounded answer ("I want a refund
        # instead", "do they have to give me a hotel?", "how do I file a claim?")
        # is answered from THAT answer, with no retrieval and no model call. It
        # sits after the clarify and unsupported branches on purpose: asking for
        # the route, and saying the carrier is not covered, both still come first.
        if allow_followup and getattr(self.s, "followup_templates", False):
            reply = self._followup(raw_question, plan, prior, history)
            if reply is not None:
                info = reply.diagnostics["followup"]
                rag_followups_total.labels(
                    template=info["template"],
                    material="quoted" if info["had_material"] else "none").inc()
                reply.text = with_openers(reply.text)
                return finish(reply, save_state=True,
                              last_answer=(prior or {}).get("last_answer"))

        # -- retrieve ----------------------------------------------------
        chunks = []
        topics = excluded_topics(question, live) if getattr(self.s, "topic_filter", False) else ()
        search_query = (retrieval_query(question, plan, live)
                        if getattr(self.s, "structured_query", False) else question)
        if intent in ("policy", "hybrid"):
            lanes = search_lanes(question, plan, live, self.s)
            res = self.retriever.retrieve(search_query, flt=flt, exclude_topics=topics, lanes=lanes,
                                          min_airline=airline_quota(question, self.s))
            timings.update(res.timings_ms)
            chunks = res.chunks
            for stage, ms in res.timings_ms.items():
                rag_stage_duration_seconds.labels(stage=stage).observe(ms / 1000)
            rag_candidates.observe(res.n_fused)
            if not chunks:
                rag_zero_result_total.inc()
            elif chunks[0].rerank_score is not None:
                rag_rerank_top_score.observe(chunks[0].rerank_score)

        # A status-only question still benefits from a little policy context,
        # because users almost always follow up with "so what am I owed?". We
        # retrieve a small amount rather than none.
        if intent == "status" and live is not None and not chunks:
            res = self.retriever.retrieve(
                "delay cancellation passenger rights compensation", top_k=3, flt=flt,
                exclude_topics=topics,
            )
            chunks = res.chunks

        # -- confidence gate --------------------------------------------
        # Retrieval can fail in a way no prompt can rescue. Detecting that here
        # is both cheaper (no wasted generation) and safer than generating from
        # weak evidence and hoping the validator catches it afterwards.
        #
        # Skipped when we have live flight data and no policy lane, because a
        # status answer does not depend on retrieval quality.
        # Recorded on every later outcome, answered ones included: calibrating the
        # thresholds needs the signals of the turns that passed, not only the gated.
        gate_record = None
        if intent in ("policy", "hybrid"):
            decision = self.gate.evaluate(question, chunks)
            gate_record = {"should_answer": decision.should_answer, "reasons": decision.reasons,
                           "signals": decision.signals}
            rag_confidence.observe(decision.confidence)
            if not decision.should_answer:
                rag_gate_abstentions_total.labels(
                    reason=gate_reason_label(decision.reasons[0] if decision.reasons else None)
                ).inc()
                rag_abstentions_total.inc()
                return finish(Answer(
                    question=question, intent=intent,
                    text=with_openers(abstention_answer(decision)),
                    assumption=plan.assumption, unsupported_airline=plan.unsupported_airline,
                    live_data=live, outcome="gated",
                    diagnostics={"gate_reasons": decision.reasons, "gate_signals": decision.signals,
                                 "gate": gate_record},
                ), save_state=True)

        # Each note is sent to the model exactly as recorded in diagnostics, so
        # evaluation can show a judge what generation was told without rebuilding it.
        prompt_notes: list[str] = []
        if plan.assumption:
            prompt_notes.append(
                f"ROUTE ASSUMPTION: {plan.assumption} The passenger could not confirm "
                "their route. Answer on this assumption; do not repeat the sentence, it is "
                "shown to them already."
            )
        if law_note := route_law_note(plan.route):
            prompt_notes.append(law_note)
        if plan.notice:
            prompt_notes.append(
                f"AIRLINE NOTICE: {plan.notice} The sources are government regulations only. "
                "Make no claim about this airline's own policies or commitments; do not repeat "
                "the notice, it is shown to them already."
            )
        if plan.flight_note:
            prompt_notes.append(
                f"FLIGHT DATA NOTE: {plan.flight_note} Do not repeat this note, it is shown "
                "to the passenger already."
                + (" The flight data does not show the delay or cancellation the passenger "
                   "mentioned: do not describe their flight as delayed or cancelled. If you "
                   "explain their rights, say they apply only if the flight is delayed or "
                   "cancelled." if plan.status_conflict else "")
            )
        prompt_question = question + "".join(f"\n\n{note}" for note in prompt_notes)
        # The whole request must fit the served context window; the sources get
        # what the prompt, history, question, a possible retry note and the
        # completion allowance leave (generation.evidence_token_budget).
        retry_reserve = "x" * ((RETRY_NOTE_MAX_CHARS + 200) * self.s.max_validation_retries)
        budget = evidence_token_budget(self.s, prompt_question, history, retry_reserve)
        ctx = build_context(chunks, live, token_budget=budget, reserve_for_answer=0)
        rag_context_tokens.observe(ctx.estimated_tokens)
        user_prompt = build_user_prompt(prompt_question, ctx, history)

        # -- generate + validate (with retries) --------------------------
        retry_count = 0
        answer_text = ""
        report = None
        base = {"question": question, "intent": intent, "assumption": plan.assumption,
                "unsupported_airline": plan.unsupported_airline, "live_data": live,
                "context_chunks": ctx.used_chunks, "built_context": ctx,
                "generation_attempted": True}
        generation_input = {"gate": gate_record, "prompt_notes": prompt_notes}
        usage: list[dict] = []

        def error_fields(error: Exception | None) -> dict:
            # The complete sanitized message and what the provider said (status, error
            # code, Retry-After, earlier retried failures); never cut short.
            if error is None:
                return {"error": None}
            details = error.details() if isinstance(error, LLMError) else {}
            return {"error": sanitize_error_text(f"{type(error).__name__}: {error}"),
                    "error_status": details.get("status"),
                    "error_code": details.get("provider_code"),
                    "error_type": details.get("provider_type"),
                    "retry_after_s": details.get("retry_after_s"),
                    "http_errors": details.get("http_errors") or []}

        def record_usage(error: Exception | None = None) -> None:
            # One record per model call, when the client reports calls (EchoLLM and
            # test doubles without `last` do not): cost and budget checks need them,
            # not the prose. A call that failed before any usage came back is still
            # recorded, with its error, so no call disappears from the accounting.
            if not hasattr(self.llm, "last"):
                return
            last = self.llm.last
            if last is None:
                if error is not None:
                    usage.append({"prompt_tokens": None, "completion_tokens": None,
                                  "reasoning_tokens": None, "reasoning_chars": None,
                                  "finish_reason": None, "truncated": False,
                                  "max_completion_tokens": getattr(self.llm, "max_tokens", None),
                                  "attempts": None, **error_fields(error)})
                return
            record = {"prompt_tokens": last.prompt_tokens,
                      "completion_tokens": last.completion_tokens,
                      "reasoning_tokens": getattr(last, "reasoning_tokens", None),
                      "reasoning_chars": getattr(last, "reasoning_chars", None),
                      "finish_reason": last.finish_reason,
                      "truncated": last.finish_reason == "length",
                      "max_completion_tokens": getattr(last, "max_completion_tokens", None),
                      "attempts": last.attempts, **error_fields(error)}
            if getattr(last, "http_errors", None):
                record["http_errors"] = last.http_errors
            usage.append(record)

        try:
            for attempt in range(self.s.max_validation_retries + 1):
                t0 = time.perf_counter()
                if hasattr(self.llm, "last"):
                    self.llm.last = None        # never report a previous call's usage
                try:
                    answer_text = self.llm.complete(SYSTEM_PROMPT, user_prompt)
                except Exception as e:
                    record_usage(e)
                    raise
                record_usage()
                timings["generate"] = (time.perf_counter() - t0) * 1000
                rag_stage_duration_seconds.labels(stage="generate").observe(
                    timings["generate"] / 1000
                )

                # The passenger's own words (not the code-written notes): restating
                # them needs no source.
                report = validate_answer(answer_text, ctx, question=question)
                if report.ok:
                    break
                for f in report.failures:
                    kind = "unknown_marker" if "not provided" in f else "uncited"
                    rag_validation_failures_total.labels(kind=kind).inc()
                if not allow_retry or attempt >= self.s.max_validation_retries:
                    break
                retry_count += 1
                rag_retries_total.inc()
                # The retry is not a bare repeat: we tell the model exactly what
                # it did wrong. A blind retry at temperature 0 would produce the
                # identical output and waste a round-trip.
                user_prompt += build_retry_note(report)
        except LLMError as e:
            return finish(Answer(
                **base, text=with_openers(LLM_ERROR_MESSAGE), retry_count=retry_count,
                outcome="llm_error",
                diagnostics={**generation_input, "usage": usage,
                             "llm_error": type(e).__name__, "llm_error_kind": e.kind,
                             "llm_error_detail": sanitize_error_text(str(e)),
                             "llm_error_status": e.status,
                             "llm_error_code": e.provider_code,
                             "llm_error_retry_after_s": e.retry_after_s},
            ), save_state=True)

        dropped_items: list[str] = []
        if not report.ok and (salvage := drop_uncited_list_items(answer_text, ctx, question)):
            # Every attempt failed, but only on bullet points: show the answer without
            # them rather than nothing. The remainder passed validation on its own.
            answer_text, report, dropped_items = salvage

        if not report.ok:
            # Never show an answer that failed validation. The passenger gets the
            # safe message; the rejected text stays in diagnostics only.
            return finish(Answer(
                **base, text=with_openers(VALIDATION_FAILURE_MESSAGE),
                validation_failures=report.failures, retry_count=retry_count,
                outcome="validation_failed",
                diagnostics={**generation_input, "usage": usage,
                             "rejected_answer": answer_text,
                             "uncited_factual_sentences": report.uncited_factual_sentences,
                             "unknown_markers": report.unknown_markers},
            ), save_state=True)

        if report.is_abstention:
            rag_abstentions_total.inc()
        return finish(Answer(
            **base, text=with_openers(answer_text), citations=report.citations,
            validation_failures=[], retry_count=retry_count,
            outcome="abstained" if report.is_abstention else "answered",
            diagnostics={**generation_input, "usage": usage,
                         "citation_coverage": report.citation_coverage,
                         "model_answer": answer_text,
                         **({"dropped_uncited_list_items": dropped_items} if dropped_items else {})},
        ), save_state=True)

    # ------------------------------------------------------------ follow-up
    def _followup(self, raw_question: str, plan: TurnPlan, prior: dict | None,
                  history: list[tuple[str, str]]) -> Answer | None:
        """
        The deterministic reply to a recognised follow-up, or None to take the
        full path.

        None is returned - and the model called - whenever anything about the
        message says it is a NEW question rather than a move within the answered
        one: no grounded answer to quote yet, the last thing said was a clarifying
        question (so this is its reply), a flight number the passenger has just
        typed, or no template matching their words. With
        followup_fallback="generate" a matched template also falls through when
        the previous answer holds no cited sentence on its topic.
        """
        record = (prior or {}).get("last_answer") or {}
        previous = record.get("text") or ""
        if not previous or extract_flight_number(raw_question):
            return None
        last_bot = next((text for role, text in reversed(history) if role == "assistant"), "")
        if is_clarification(last_bot):
            return None
        # A message that names a route AND a disruption is a new situation, not a
        # move within the answered one - "my United flight from London was delayed
        # 5 hours" would otherwise be swallowed by a template and answered from an
        # answer about a different flight. Either alone is fine: "does it matter
        # that I was flying from the UK?" names a place and no disruption, "I don't
        # know why it was cancelled" the reverse.
        named = disruptions_named(raw_question)
        if _names_route(detect_jurisdictions(raw_question)) and named:
            return None
        # A disruption the answered turn was not about is a new question on the same
        # flight: "How much compensation for denied boarding?" after a cancellation
        # answer must be answered (route and carrier still carry over), not given the
        # cancellation sentences back. An answer saved before this field existed
        # cannot say what it covered, so any named disruption takes the full path.
        answered = record.get("disruptions")
        if named and (answered is None or not set(named) <= set(answered)):
            return None
        spec = followups.match(raw_question)
        if spec is None:
            return None
        code = (plan.flt.get("airline_scope") or (prior or {}).get("airline") or "")
        name = AIRLINE_NAMES.get(str(code).upper(), "the airline")
        reply = followups.compose(spec, previous, name)
        if not reply.had_material and getattr(self.s, "followup_fallback", "template") == "generate":
            return None
        return Answer(
            question=raw_question, intent=plan.intent, text=reply.text,
            citations=followup_citations(record, reply.markers),
            live_data=plan.live if plan.live_status_usable else None,
            assumption=plan.assumption, unsupported_airline=plan.unsupported_airline,
            outcome="followup", generation_attempted=False,
            diagnostics={"followup": {"template": spec.id, "had_material": reply.had_material,
                                      "markers": reply.markers, "quoted": reply.quoted}},
        )

    def _save_turn(self, conversation_id, question, answer_text, citations) -> None:
        # The EFFECTIVE question is saved, so a second clarification round
        # still has the passenger's original situation to append to.
        if not conversation_id or self.store is None:
            return
        with contextlib.suppress(Exception):
            self.store.save_message(conversation_id, "user", question)
            self.store.save_message(conversation_id, "assistant", answer_text, citations)

    def _save_state(self, conversation_id, plan: TurnPlan, last_answer: dict | None = None) -> None:
        if not conversation_id or self.store is None or not hasattr(self.store, "save_state"):
            return
        with contextlib.suppress(Exception):
            self.store.save_state(conversation_id, turn_state(plan, last_answer))


def build_pipeline(settings, store=None):
    """
    Wire everything together from config.

    Kept as one function so there is exactly one place where the object graph is
    constructed - the API, the eval harness, and the CLI all call this, which
    means they cannot drift into testing different configurations.
    """
    from .embeddings import build_embedder
    from .generation import build_llm
    from .retrieval import HybridRetriever, build_reranker
    from .store import PgVectorStore
    from .tools import build_tool

    store = store or PgVectorStore(settings.pg_dsn, hnsw_ef_search=settings.hnsw_ef_search)
    embedder = build_embedder(settings)
    reranker = build_reranker(settings)
    retriever = HybridRetriever(store, embedder, reranker, settings)
    llm = build_llm(settings)
    tool = build_tool(settings, store=store)
    return RagPipeline(retriever, llm, tool, settings, store=store)
