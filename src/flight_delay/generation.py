"""
Generation: context assembly, prompting, LLM client, and post-hoc validation.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .ingest import estimate_tokens
from .models import Chunk, Citation, FlightStatus

# ==========================================================================
# Prompt
# ==========================================================================

# The prompt lives in system_prompt.md at the repository root
_REPO_PROMPT = Path(__file__).resolve().parents[2] / "system_prompt.md"
SYSTEM_PROMPT_PATH = Path(os.environ.get("SYSTEM_PROMPT_PATH") or _REPO_PROMPT)


def load_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RuntimeError(
            f"System prompt not found at {path}. It is the canonical production prompt; "
            "ship system_prompt.md with the app (the Dockerfile copies it to /app) or set "
            "SYSTEM_PROMPT_PATH to its location."
        ) from None
    if not text:
        raise RuntimeError(f"System prompt at {path} is empty.")
    return text


SYSTEM_PROMPT = load_system_prompt()
SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT_PATH.read_bytes()).hexdigest()[:16]


_SOURCE_KIND_LABEL = {
    "regulation": "LAW — binding regulation",
    "guidance": "LAW — regulator's official guidance on the regulation",
    "contract": "AIRLINE CONTRACT — contractually binding on the carrier",
    "service_plan": "AIRLINE COMMITMENT — carrier's own published promise",
    "policy": "AIRLINE POLICY — carrier's published guidance",
}


def _format_flight_block(fs: FlightStatus) -> str:
    """Render live status compactly."""
    lines = [f"Flight: {fs.flight_iata}"]
    if fs.airline_iata:
        lines.append(f"Airline: {fs.airline_iata}")
    if fs.is_codeshare:
        lines.append(f"Operated by: {fs.operating_airline_iata}"
                     + (f" as {fs.operating_flight_iata}" if fs.operating_flight_iata else "")
                     + " (codeshare)")
    if fs.status:
        lines.append(f"Status: {fs.status}")
    if fs.dep_iata:
        lines.append(f"Departure: {fs.dep_iata}")
    if fs.dep_time:
        lines.append(f"Scheduled departure: {fs.dep_time}")
    if fs.dep_estimated:
        lines.append(f"Estimated departure: {fs.dep_estimated}")
    if fs.dep_delayed is not None:
        lines.append(f"Departure delay: {fs.dep_delayed} minutes")
    if fs.arr_iata:
        lines.append(f"Arrival: {fs.arr_iata}")
    if fs.arr_time:
        lines.append(f"Scheduled arrival: {fs.arr_time}")
    if fs.arr_estimated:
        lines.append(f"Estimated arrival: {fs.arr_estimated}")
    if fs.arr_delayed is not None:
        lines.append(f"Arrival delay: {fs.arr_delayed} minutes")
    lines.append("Delay cause: NOT REPORTED BY THIS DATA SOURCE")
    return "\n".join(lines)


_FLIGHT_HEADERS = {
    "live": "FLIGHT DATA (real-time; not a citable source)",
    "cache": "FLIGHT DATA (real-time, cached briefly; not a citable source)",
    "fixture": "FLIGHT DATA (RECORDED on an earlier date, NOT current; not a citable source)",
    "synthetic": "FLIGHT DATA (SYNTHETIC test record, not a real flight; not a citable source)",
}


def flight_block(fs: FlightStatus) -> str:
    """The labelled flight block exactly as the model sees it (without trailing blank line)."""
    header = _FLIGHT_HEADERS.get(fs.source, _FLIGHT_HEADERS["live"])
    return f"{header}:\n{_format_flight_block(fs)}"


@dataclass
class BuiltContext:
    text: str
    source_map: dict[str, Chunk]  # "S1" -> Chunk
    used_chunks: list[Chunk]
    estimated_tokens: int
    flight_text: str = ""         # the FLIGHT DATA block, "" when there is none


def section_path(ch: Chunk) -> str:
    """The breadcrumb without its leading document title, which the source header prints already"""
    return ch.breadcrumb.removeprefix(f"{ch.doc_title} > ") or ch.breadcrumb


def build_context(
    chunks: list[Chunk],
    live: FlightStatus | None,
    *,
    token_budget: int = 6000,
    reserve_for_answer: int = 900,
) -> BuiltContext:
    """
    Assemble the SOURCES block within a token budget.

    """
    available = token_budget - reserve_for_answer
    flight_text = ""
    if live is not None:
        flight_text = f"{flight_block(live)}\n\n"
        available -= len(flight_text) // 4

    def cost_of(ch: Chunk) -> int:
        return estimate_tokens(ch.text) + 40  

    # Guaranteed chunks first, in their existing relevance order, so the budget
    # is committed to them before relevance spends it. 
    kept: list[Chunk] = []
    used = 0
    for ch in chunks:
        if not getattr(ch, "guaranteed", False):
            continue
        cost = cost_of(ch)
        if used + cost > available:
            continue
        kept.append(ch)
        used += cost

    # Then the rest, best-first, keeping the strongest that still fit.
    for ch in chunks:
        if getattr(ch, "guaranteed", False):
            continue
        cost = cost_of(ch)
        if used + cost > available:
            continue
        kept.append(ch)
        used += cost

    # Emit weakest -> strongest ,so the best is adjacent to the question. `chunks`
    # arrives best-first, so restore that order before reversing.
    rank = {id(ch): i for i, ch in enumerate(chunks)}
    kept.sort(key=lambda c: rank[id(c)])
    ordered = list(reversed(kept))

    source_map: dict[str, Chunk] = {}
    blocks: list[str] = []
    for i, ch in enumerate(ordered, start=1):
        marker = f"S{i}"
        source_map[marker] = ch
        eff = f" | effective {ch.effective_date}" if ch.effective_date else ""
     
        kind = _SOURCE_KIND_LABEL.get(
            ch.doc_type,
            "LAW" if ch.source_class == "government" else "AIRLINE POLICY",
        )
        blocks.append(
            f"[{marker}] ({kind}) {ch.doc_title} — {section_path(ch)}"
            f" | jurisdiction: {ch.jurisdiction}{eff}\n{ch.text}"
        )

    text = flight_text + "SOURCES:\n\n" + "\n\n---\n\n".join(blocks)
    return BuiltContext(
        text=text,
        source_map=source_map,
        used_chunks=ordered,
        estimated_tokens=len(text) // 4,
        flight_text=flight_text.strip(),
    )


_HISTORY_HEADER = "EARLIER IN THIS CONVERSATION:\n"
_ANSWER_INSTRUCTION = "Answer using only the sources above, with [S] markers."
# The fixed text build_user_prompt() and build_context() add around the variable
# parts, so evidence_token_budget() can count it.
_PROMPT_SCAFFOLD = _HISTORY_HEADER + "SOURCES:\n\n" + "\nQUESTION: \n\n" + _ANSWER_INSTRUCTION


def build_user_prompt(question: str, context: BuiltContext, history: list[tuple[str, str]] | None = None) -> str:
    parts = []
    if history:
        convo = "\n".join(f"{r.upper()}: {c[:400]}" for r, c in history[-4:])
        parts.append(f"{_HISTORY_HEADER}{convo}\n")
    parts.append(context.text)
    parts.append(f"\nQUESTION: {question}\n\n{_ANSWER_INSTRUCTION}")
    return "\n".join(parts)


# ==========================================================================
# Validation
# ==========================================================================

_MARKER_RE = re.compile(r"\[(S\d+)\]")
_MARKER_GROUP_RE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)+)\s*\]")


def normalize_citation_markers(text: str) -> str:
    """Rewrite "[S5, S6]" as "[S5][S6]"; leave every other bracket untouched."""
    return _MARKER_GROUP_RE.sub(
        lambda m: "".join(f"[{part}]" for part in re.split(r"\s*[,;]\s*", m.group(1))), text)
# A sentence is treated as "factual" if it contains a digit, a currency amount,
# a section symbol, or a modal obligation verb.
_FACTUAL_HINT = re.compile(
    r"(\d|\$|§|\bmust\b|\bshall\b|\bentitled\b|\brequired\b|\bcompensat|\bliable\b)", re.I
)
# Words that make a sentence a claim about RIGHTS or MONEY. A sentence carrying one
# is never excused as a mere restatement of the flight data.
_LEGAL_HINT = re.compile(
    r"(€|£|\$|§|\bmust\b|\bshall\b|\bentitle|\brequired\b|\bcompensat|\bliable\b|"
    r"\brefund|\bowe[ds]?\b|\bright(s)?\b|\breimburse|\bvoucher|\bhotel|\bmeal|\bregulation|"
    r"\barticle\b|\bcfr\b|\b261\b|\bpercent|%)", re.I
)

_SRC = r"(?:the |these |my )?(?:provided |supplied |retrieved |available |above )?(?:sources?|documents?)"
_SRC_TAIL = (r"(?: (?:provided|supplied|retrieved|above|given|available|I have|I was given"
             r"|you (?:provided|supplied|gave)))?")
_COVER_VERB = (r"(?:say|state|specify|cover|address|contain|include|mention|discuss|provide|give|"
               r"answer|describe|list)")
# Past participles of the same verbs, for the passive phrasing 
_COVERED_IN = (r"(?:specified|stated|given|listed|provided|set out|spelled out|described|"
               r"mentioned|included|addressed|covered (?:in|by)|found in)")
_ABSTENTION_HINT = re.compile(
    r"(" + _SRC + _SRC_TAIL + r"(?: (?:do|does|did))? ?(?:not|n't|never) " + _COVER_VERB
    + r"|\bnone of " + _SRC + _SRC_TAIL + r" " + _COVER_VERB + r"(?:s|es)?"
    + r"|\bnot (?:in |" + _COVERED_IN + r" (?:in |by )?)" + _SRC
    + r"|\b(?:no|not enough|insufficient) (?:source|information) (?:in the (?:provided )?sources? )?"
      r"(?:covers?|address(?:es)?|about|on|to)"
    + r"|\bI (?:have|found|was given) no (?:source|information)"
    + r"|\b(?:cannot|can't|can not|am unable to|unable to) (?:answer|determine)"
    + r"|\binsufficient information)", re.I
)
_NUMBER_RE = re.compile(r"\d+(?:[.,:]\d+)*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Full stops that do NOT end a sentence
_ABBREV_END = re.compile(
    r"(?:\b[A-Z](?:\.[A-Z])+\.|\b(?:[Ee]\.g|[Ii]\.e|etc|vs|[Nn]o|[Nn]os|[Aa]rt|[Ss]ec|"
    r"[Pp]ara|[Pp]t|approx|[Mm]r|[Mm]rs|[Mm]s|[Dd]r)\.)$")


def _split_sentences(text: str) -> list[str]:
    """Split on sentence enders, then re-join a piece cut after an abbreviation."""
    parts: list[str] = []
    for piece in _SENTENCE_SPLIT.split(text):
        if parts and _ABBREV_END.search(parts[-1]):
            parts[-1] += " " + piece
        else:
            parts.append(piece)
    return parts
# Clause boundaries a decline can hide a claim behind ("..., but you are owed €600").
_CLAUSE_SPLIT = re.compile(r"\s*(?:;|—|,?\s+\b(?:but|however|although|though|whereas|yet)\b,?)\s+", re.I)
# The regimes' own names carry a number that is not a claim ("under UK261").
_REGIME_NAME = re.compile(r"\b(?:EU|UK)\s?261(?:/2004)?\b|\bRegulation\s+\(EC\)\s+No\.?\s*261/2004", re.I)
# What makes a sentence state a specific amount or reference, wherever it sits.
_AMOUNT_HINT = re.compile(r"[€£$§%]|\b(?:EUR|GBP|USD|CAD|euros?|pounds?|dollars?|percent)\b", re.I)
_DEADLINE_HINT = re.compile(r"\b(?:within|no later than|deadline|by the end of)\b", re.I)
# In a recommendation's main clause, these turn advice into a policy statement.
_ASSERTION_HINT = re.compile(
    r"\b(?:must|shall|required|liable|entitled|owe[ds]?|guarantee[ds]?|"
    r"will (?:pay|refund|reimburse|provide|give|owe|compensate|cover|rebook))\b", re.I)
# Words that make a sentence carrier POLICY rather than a restatement of the question.
_POLICY_HINT = re.compile(
    r"\b(?:rebook|reroute|voucher|credit|miles|eligib|polic|commit|guarantee|offer|provide|pay|"
    r"cash|waive|fee|within|appl(?:y|ies)|cover)", re.I)
# A referral: the passenger is told whom to contact or what to do.
_LEADING_CLAUSE = re.compile(
    r"^(?:if|to|when|once|before|after|for|in order to|should)\b[^,]{0,200},\s*", re.I)
_ADVICE_OPENER = re.compile(
    r"^(?:please\s+|you (?:may|can|could|should|might)(?: (?:want|wish|need) to)?\s+|"
    r"(?:we|I) (?:recommend|suggest)(?: that you)?\s+|consider\s+)", re.I)
_ADVICE_VERB = re.compile(
    r"^(?:contact|consult|ask|check|call|visit|email|write to|reach out|speak|talk|file|submit|"
    r"keep|save|request|report|see|refer|review|confirm|lodge|escalate|complain|apply)\b", re.I)
# Only words of 3+ letters are compared (_WORD_RE), so shorter ones need no entry.
_STOPWORDS = frozenset({
    "the", "and", "for", "from", "with", "was", "were", "are", "been", "your", "you", "our",
    "this", "that", "its", "because", "due", "has", "have", "had", "which", "who", "what", "when"})
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_HEADING_LINE = re.compile(r"^(?:#{1,6}\s+(?P<hash>.+?)|(?P<em>\*\*|__)(?P<bold>.+?)(?P=em):?)\s*$")
_HEADING_AMOUNT = re.compile(
    r"[€£$§%]|\d+\s*(?:percent|days?|hours?|minutes?|weeks?|months?|euros?|pounds?|dollars?|"
    r"EUR|GBP|USD|CAD)\b", re.I)


@dataclass
class ValidationReport:
    """
    What the validator can and cannot establish.
    """

    ok: bool
    failures: list[str]
    citations: list[Citation]
    unknown_markers: list[str]
    uncited_factual_sentences: list[str]
    is_abstention: bool
    factual_sentences: int = 0         
    cited_factual_sentences: int = 0    
    abstention_sentences: int = 0
    flight_data_sentences: int = 0      
    user_fact_sentences: int = 0        
    recommendation_sentences: int = 0   

    @property
    def citation_coverage(self) -> float | None:
        """Share of factual sentences that carry a marker; None when there were none."""
        if not self.factual_sentences:
            return None
        return self.cited_factual_sentences / self.factual_sentences


def _restates_flight_data(sentence: str, flight_text: str) -> bool:
    """
    True for a sentence that only reports the FLIGHT DATA block 
    """
    if not flight_text or _LEGAL_HINT.search(sentence):
        return False
    available = set(_NUMBER_RE.findall(flight_text))
    return all(n in available for n in _NUMBER_RE.findall(sentence))


def _unknown_numbers(text: str, known: set[str]) -> bool:
    """True when the text states a number the passenger did not give (regime names excluded)."""
    return any(n not in known for n in _NUMBER_RE.findall(_REGIME_NAME.sub("", text)))


def _is_source_decline(sentence: str, known_numbers: set[str]) -> bool:
    """
    A sentence that declines on SOURCE COVERAGE
    """
    clauses = [c for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]
    declines = [c for c in clauses if _ABSTENTION_HINT.search(c)]
    if not declines:
        return False
    for clause in clauses:
        if clause in declines:
            rest = _ABSTENTION_HINT.sub("", clause)
            if _AMOUNT_HINT.search(rest) or _unknown_numbers(rest, known_numbers):
                return False
        elif (_AMOUNT_HINT.search(clause) or _unknown_numbers(clause, known_numbers)
              or (len(clause.strip()) >= 25 and _FACTUAL_HINT.search(clause))):
            # An amount or a number the passenger never gave disqualifies the
            # decline at any length: "..., but you are owed 600 euros" is 23 chars.
            return False
    return True


def _restates_user_facts(sentence: str, question: str) -> bool:
    """
    True for a sentence that only repeats what the PASSENGER said 
    """
    if not question or _LEGAL_HINT.search(sentence) or _POLICY_HINT.search(sentence):
        return False
    if _unknown_numbers(sentence, set(_NUMBER_RE.findall(question))):
        return False
    words = [w for w in (m.lower() for m in _WORD_RE.findall(sentence)) if w not in _STOPWORDS]
    asked = {m.lower() for m in _WORD_RE.findall(question)}
    return bool(words) and sum(w in asked for w in words) / len(words) >= 0.6


def _is_follow_up_question(sentence: str, known_numbers: set[str]) -> bool:
    """
    A question back to the passenger
    """
    s = sentence.strip()
    if not s.endswith("?"):
        return False
    if _AMOUNT_HINT.search(s) or _DEADLINE_HINT.search(s) or _unknown_numbers(s, known_numbers):
        return False
    return not _ASSERTION_HINT.search(s)


def _is_recommendation(sentence: str, known_numbers: set[str]) -> bool:
    """
    
    """
    lead = _LEADING_CLAUSE.match(sentence)
    core = sentence[lead.end():] if lead else sentence
    core = _ADVICE_OPENER.sub("", core.strip())
    if not _ADVICE_VERB.match(core):
        return False
    if _AMOUNT_HINT.search(sentence) or _DEADLINE_HINT.search(sentence):
        return False
    if _unknown_numbers(sentence, known_numbers):
        return False
    return not (_ASSERTION_HINT.search(core) or _CLAUSE_SPLIT.search(core))


def _answer_sentences(answer: str) -> list[str]:
    units: list[str] = []
    buf = ""
    for raw in answer.splitlines():
        line = raw.strip()
        heading = _HEADING_LINE.match(line) if line else None
        if heading:
            label = _LIST_MARKER.sub("", heading.group("hash") or heading.group("bold") or "")
            if (not re.search(r"[.!?]$", label) and len(label.split()) <= 12
                    and not _MARKER_RE.search(label) and not _HEADING_AMOUNT.search(label)):
                units.append(buf)
                buf = ""
                continue
        item = _LIST_MARKER.match(line)
        if not line or item:
            units.append(buf)
            buf = line[item.end():] if item else ""
        elif buf and not re.search(r"[.!?:][\"')\]*_]*$", buf):
            buf += " " + line
        else:
            units.append(buf)
            buf = line
    units.append(buf)
    sentences = []
    for unit in units:
        text = re.sub(r"\*\*|__", "", unit)
        text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
        sentences.extend(s.strip() for s in _split_sentences(text.strip()) if s.strip())
    return sentences


def validate_answer(answer: str, context: BuiltContext, question: str = "") -> ValidationReport:
    """
    Check the model's output against the sources it was given, sentence by sentence.

    """
    failures: list[str] = []
    citations: list[Citation] = []
    unknown: list[str] = []

    # LLMClient normalises already; doing it again keeps the validator correct for
    # any other caller, and costs one no-op regex pass.
    answer = normalize_citation_markers(answer)
    found = _MARKER_RE.findall(answer)
    for marker in dict.fromkeys(found):
        ch = context.source_map.get(marker)
        if ch is None:
            unknown.append(marker)
            continue
        citations.append(
            Citation(
                marker=marker,
                chunk_id=ch.chunk_id,
                doc_title=ch.doc_title,
                section_id=ch.section_id,
                source_url=ch.source_url,
            )
        )

    if unknown:
        failures.append(f"cites sources that were not provided: {', '.join(unknown)}")

    uncited: list[str] = []
    factual = cited = abstaining = flight_only = user_facts = advice = 0
    known_numbers = set(_NUMBER_RE.findall(question or ""))
    for s in _answer_sentences(answer):
        if _is_source_decline(s, known_numbers):
            # Declines and claims nothing else. "The sources do not specify this,
            # but you are entitled to €600" still needs its marker.
            abstaining += 1
            continue
        if len(s) < 25 or not _FACTUAL_HINT.search(s):
            continue
        if not _MARKER_RE.search(s) and _is_follow_up_question(s, known_numbers):
            advice += 1
            continue
        if _MARKER_RE.search(s):
            factual += 1
            cited += 1
        elif _restates_flight_data(s, context.flight_text):
            flight_only += 1
        elif _restates_user_facts(s, question):
            user_facts += 1
        elif _is_recommendation(s, known_numbers):
            advice += 1
        else:
            factual += 1
            uncited.append(s)
    if uncited:
        failures.append(f"{len(uncited)} factual sentence(s) carry no citation")

    # A decline with no claim needing a source is a valid answer on its own.
    is_abstention = abstaining > 0 and not found and not uncited
    if not found and not is_abstention and (factual or not flight_only):
        failures.append("no citations at all and not an explicit abstention")

    return ValidationReport(
        ok=not failures,
        failures=failures,
        citations=citations,
        unknown_markers=unknown,
        uncited_factual_sentences=uncited,
        is_abstention=is_abstention,
        factual_sentences=factual,
        cited_factual_sentences=cited,
        abstention_sentences=abstaining,
        flight_data_sentences=flight_only,
        user_fact_sentences=user_facts,
        recommendation_sentences=advice,
    )


def drop_uncited_list_items(answer: str, context: BuiltContext, question: str = ""
                            ) -> tuple[str, ValidationReport, list[str]] | None:
    
    first = validate_answer(answer, context, question=question)
    if first.ok or first.unknown_markers or not first.uncited_factual_sentences:
        return None
    bad = set(first.uncited_factual_sentences)
    kept: list[str] = []
    dropped: list[str] = []
    for line in normalize_citation_markers(answer).splitlines():
        if _LIST_MARKER.match(line) and bad & set(_answer_sentences(line)):
            dropped.append(line.strip())
        else:
            kept.append(line)
    if not dropped:
        return None
    text = "\n".join(kept).strip()
    report = validate_answer(text, context, question=question)
    if not report.ok or report.is_abstention or not report.cited_factual_sentences:
        return None
    return text, report, dropped


# ==========================================================================
# LLM client
# ==========================================================================


_SECRET_PATTERNS = (
    (re.compile(r"\bgsk_[A-Za-z0-9]{8,}"), "gsk_<redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "sk-<redacted>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer <redacted>"),
    (re.compile(r"(?i)(api[_-]?key[\"']?\s*[=:]\s*[\"']?)[^\s\"',}]+"), r"\1<redacted>"),
    # the provider account the key belongs to (Groq names it in rate-limit errors)
    (re.compile(r"\borg_[A-Za-z0-9]+"), "org_<redacted>"),
)


def sanitize_error_text(text: str, secrets: tuple[str, ...] = ()) -> str:
    """
    The COMPLETE error text with keys and account ids removed. Never truncated: a
    provider's explanation (which limit, what was requested) is the diagnosis.
    """
    out = str(text)
    for secret in secrets:
        if secret and len(secret) >= 8:
            out = out.replace(secret, "<redacted>")
    for pattern, replacement in _SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


class LLMError(RuntimeError):
    """
    A generation call did not produce a usable answer
    """

    kind = "error"

    def __init__(self, message: str, *, status: int | None = None, provider_code: str | None = None,
                 provider_type: str | None = None, provider_message: str | None = None,
                 retry_after_s: float | None = None, http_errors: list[dict] | None = None):
        super().__init__(message)
        self.status = status
        self.provider_code = provider_code
        self.provider_type = provider_type
        self.provider_message = provider_message
        self.retry_after_s = retry_after_s
        self.http_errors = list(http_errors or [])

    def details(self) -> dict:
        return {"status": self.status, "provider_code": self.provider_code,
                "provider_type": self.provider_type, "provider_message": self.provider_message,
                "retry_after_s": self.retry_after_s, "http_errors": self.http_errors}


class LLMTransientError(LLMError):
    """Rate limit, server error, timeout or dropped connection, still failing after retries."""

    kind = "transient"


class LLMRequestError(LLMError):
    """The provider rejected the request itself (bad key, unknown model, prompt too long)."""

    kind = "request"


class LLMResponseError(LLMError):
    """The provider answered, but not with a complete answer (truncated, empty, malformed)."""

    kind = "response"


@dataclass
class Completion:
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    attempts: int
    # The part of completion_tokens spent reasoning, when the provider reports it.
    reasoning_tokens: int | None = None
    # Characters of reasoning text returned (message.reasoning / reasoning_content or
    # an inline <think> block): evidence of reasoning when token details are absent.
    reasoning_chars: int = 0
    # The completion cap this call was sent with.
    max_completion_tokens: int | None = None
    # Retried HTTP failures before this success (LLMError.details() fields + waited_s).
    http_errors: list[dict] | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S | re.I)


def _strip_reasoning(text: str) -> str:
    """
    Remove a reasoning model's <think> block. Some OpenAI-compatible servers and
    hosted models return the model's deliberation inline in the content; it is not
    an answer, and its sentences would be validated and shown.
    """
    return _THINK_RE.sub("", text).strip()


class LLMClient:
    """
    Talks to any OpenAI-compatible /chat/completions endpoint.
    """

    RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    def __init__(self, base_url: str, model: str, api_key: str = "x",
                 max_tokens: int = 700, temperature: float = 0.0, timeout: float = 120.0,
                 *, provider: str = "openai-compatible", max_retries: int = 3,
                 retry_max_wait_s: float = 30.0, extra_body: dict | None = None,
                 price_input_per_mtok: float | None = None,
                 price_output_per_mtok: float | None = None,
                 min_call_interval_s: float = 0.0,
                 transport: httpx.BaseTransport | None = None,
                 sleep=time.sleep, clock=time.monotonic):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        
        self.temperature = temperature
        self.timeout = timeout
        self.provider = provider
        self.max_retries = max_retries
        self.retry_max_wait_s = retry_max_wait_s
        self.extra_body = dict(extra_body or {})
        self.cap_parameter = "max_completion_tokens" if provider == "groq" else "max_tokens"
        self.price_input_per_mtok = price_input_per_mtok
        self.price_output_per_mtok = price_output_per_mtok
        self.min_call_interval_s = min_call_interval_s
        self._transport = transport
        self._sleep = sleep
        self._clock = clock
        self._last_request_start: float | None = None
        self.last: Completion | None = None

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _backoff_s(self, attempt: int) -> float:
        return min(self.retry_max_wait_s, (2 ** attempt) * (0.5 + random.random() / 2))

    def _pace(self) -> None:
        """Wait until min_call_interval_s has passed since the previous request started."""
        if self.min_call_interval_s and self._last_request_start is not None:
            gap = self.min_call_interval_s - (self._clock() - self._last_request_start)
            if gap > 0:
                self._sleep(gap)
        self._last_request_start = self._clock()

    def _http_error(self, response: httpx.Response) -> dict:
        """What the provider said about a failed request, sanitized and complete."""
        code = kind = None
        message = response.text
        try:
            error = response.json().get("error")
            if isinstance(error, dict):
                code, kind = error.get("code"), error.get("type")
                message = error.get("message") or message
            elif isinstance(error, str):
                message = error
        except (ValueError, AttributeError):
            pass
        retry_after = None
        header = response.headers.get("retry-after")
        if header:
            try:
                retry_after = max(0.0, float(header))
            except ValueError:
                retry_after = None
        return {"status": response.status_code,
                "provider_code": None if code is None else str(code),
                "provider_type": None if kind is None else str(kind),
                "provider_message": sanitize_error_text(message, (self.api_key,)),
                "retry_after_s": retry_after}

    @staticmethod
    def _request_too_large(err: dict) -> bool:
        return (err["status"] == 413 or err["provider_code"] == "request_too_large"
                or (err["provider_message"] or "").lstrip().lower().startswith("request too large"))

    def generate(self, system: str, user: str) -> Completion:
        from .metrics import llm_calls_total, llm_retries_total

        payload = {
            **self.extra_body,
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Groq deprecates max_tokens; vLLM and other servers document max_tokens.
            self.cap_parameter: self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        attempts = 0
        last_problem = ""
        history: list[dict] = []
        with httpx.Client(timeout=self.timeout, transport=self._transport) as c:
            while True:
                attempts += 1
                response = None
                err: dict = {}
                self._pace()
                try:
                    response = c.post(f"{self.base_url}/chat/completions", json=payload,
                                      headers=self._headers())
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_problem = sanitize_error_text(f"{type(e).__name__}: {e}", (self.api_key,))
                    reason = "timeout" if isinstance(e, httpx.TimeoutException) else "connection"
                else:
                    if response.status_code < 400:
                        break
                    err = self._http_error(response)
                    described = (f"HTTP {err['status']}"
                                 + (f" [{err['provider_code']}]" if err["provider_code"] else "")
                                 + f": {err['provider_message']}")
                    if response.status_code not in self.RETRYABLE_STATUS or self._request_too_large(err):
                        llm_calls_total.labels(provider=self.provider, outcome="request_error").inc()
                        raise LLMRequestError(
                            f"{self.provider} rejected the request: {described}"
                            + (" (not retried: the request exceeds a provider limit)"
                               if self._request_too_large(err) else ""),
                            **err, http_errors=history)
                    last_problem = described
                    reason = "rate_limited" if response.status_code == 429 else "server_error"
                if attempts > self.max_retries:
                    llm_calls_total.labels(provider=self.provider, outcome="transient_error").inc()
                    raise LLMTransientError(
                        f"{self.provider} still failing after {attempts} attempts: {last_problem}",
                        **err, http_errors=history)
                retry_after = err.get("retry_after_s")
                if retry_after is not None and retry_after > self.retry_max_wait_s:
                    llm_calls_total.labels(provider=self.provider, outcome="transient_error").inc()
                    raise LLMTransientError(
                        f"{self.provider} asked to retry after {retry_after:g}s (Retry-After), more "
                        f"than retry_max_wait_s={self.retry_max_wait_s:g}; not retried early: "
                        f"{last_problem}", **err, http_errors=history)
                wait = retry_after if retry_after is not None else self._backoff_s(attempts)
                history.append({**(err or {"status": None, "provider_code": None, "provider_type": None,
                                           "provider_message": last_problem, "retry_after_s": None}),
                                "waited_s": round(wait, 3)})
                llm_retries_total.labels(provider=self.provider, reason=reason).inc()
                self._sleep(wait)

        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
            finish_reason = choice.get("finish_reason")
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            reasoning_chars = (len(reasoning) if isinstance(reasoning, str) else 0) + sum(
                len(m) for m in _THINK_RE.findall(content or ""))
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            llm_calls_total.labels(provider=self.provider, outcome="malformed").inc()
            raise LLMResponseError(f"{self.provider} returned an unreadable body: {e}") from e

        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        completion = Completion(
            text=normalize_citation_markers(_strip_reasoning(content or "")),
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            attempts=attempts,
            reasoning_tokens=details.get("reasoning_tokens") if isinstance(details, dict) else None,
            reasoning_chars=reasoning_chars,
            max_completion_tokens=self.max_tokens,
            http_errors=history,
        )
        self.last = completion
        self._record_usage(completion)

        if finish_reason == "length":
            llm_calls_total.labels(provider=self.provider, outcome="truncated").inc()
            raise LLMResponseError(
                f"answer truncated at {self.cap_parameter}={self.max_tokens} (finish_reason=length; "
                f"reasoning tokens count toward it)")
        if not completion.text:
            llm_calls_total.labels(provider=self.provider, outcome="empty").inc()
            raise LLMResponseError(f"{self.provider} returned no answer text "
                                   f"(finish_reason={finish_reason!r})")
        llm_calls_total.labels(provider=self.provider, outcome="ok").inc()
        return completion

    def _record_usage(self, completion: Completion) -> None:
        """Token counters when the provider reports usage; cost only when prices are configured."""
        from .metrics import rag_query_cost_usd, rag_tokens_total

        if completion.prompt_tokens is not None:
            rag_tokens_total.labels(kind="prompt").inc(completion.prompt_tokens)
        if completion.completion_tokens is not None:
            rag_tokens_total.labels(kind="completion").inc(completion.completion_tokens)
        if (self.price_input_per_mtok is not None and self.price_output_per_mtok is not None
                and completion.prompt_tokens is not None
                and completion.completion_tokens is not None):
            rag_query_cost_usd.observe(
                completion.prompt_tokens * self.price_input_per_mtok / 1e6
                + completion.completion_tokens * self.price_output_per_mtok / 1e6)

    def complete(self, system: str, user: str) -> str:
        return self.generate(system, user).text


class EchoLLM:


    provider = "echo"

    def __init__(self, *a, **kw):
        self.last: Completion | None = None

    def generate(self, system: str, user: str) -> Completion:  # noqa: ARG002
        m = re.search(r"\[(S\d+)\][^\n]*\n(.{0,300})", user, re.S)
        text = ("The provided sources do not cover this question." if not m else
                f"Based on the retrieved policy text, the relevant provision states the "
                f"following [{m.group(1)}].")
        self.last = Completion(text, "stop", None, None, 1)
        return self.last

    def complete(self, system: str, user: str) -> str:
        return self.generate(system, user).text


def build_llm(settings):
    """The generator named by LLM_PROVIDER. Never falls back to a stand-in silently."""
    if settings.llm_provider == "echo" or settings.llm_base_url == "echo":
        if not settings.allow_test_doubles:
            raise RuntimeError("the echo generator is a test double: set ALLOW_TEST_DOUBLES=true "
                               "for an offline smoke run, or configure a real LLM_PROVIDER")
        return EchoLLM()
    if not settings.llm_base_url or settings.llm_base_url == "none":
        raise RuntimeError("LLM_BASE_URL is empty: configure LLM_PROVIDER, LLM_BASE_URL and "
                           "LLM_MODEL explicitly (there is no default fallback generator)")
    return LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        max_tokens=settings.completion_cap_tokens,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        provider=settings.llm_provider,
        max_retries=settings.llm_max_retries,
        retry_max_wait_s=settings.llm_retry_max_wait_s,
        extra_body=settings.llm_extra_body,
        price_input_per_mtok=settings.llm_price_input_per_mtok,
        price_output_per_mtok=settings.llm_price_output_per_mtok,
        min_call_interval_s=settings.llm_min_call_interval_s,
    )


# ==========================================================================
# Request budget
# ==========================================================================

REQUEST_SAFETY_MARGIN = 0.10
_TEMPLATE_OVERHEAD_TOKENS = 64


def evidence_token_budget(settings, question: str, history: list[tuple[str, str]] | None,
                          extra_prompt_text: str = "") -> int:
    """
    Tokens left for the SOURCES block (sources + flight data) in one request.

   
    """
    history_text = ""
    if history:
        history_text = "\n".join(f"{r.upper()}: {c[:400]}" for r, c in history[-4:])
    fixed = (len(SYSTEM_PROMPT) + len(history_text) + len(question) + len(extra_prompt_text)
             + len(_PROMPT_SCAFFOLD)) // 4 + _TEMPLATE_OVERHEAD_TOKENS
    usable = int(settings.llm_context_window * (1 - REQUEST_SAFETY_MARGIN))
    return max(0, min(settings.context_token_budget,
                      usable - fixed - settings.context_answer_reserve_tokens))
