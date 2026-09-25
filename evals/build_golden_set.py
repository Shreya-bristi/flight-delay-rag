#!/usr/bin/env python3
"""
Build evals/golden_set.jsonl from the case specs in evals/golden/cases_*.py.

OUTPUT FIELDS (per line)
    id, question, answerable                  read by both evaluation stages
    scenario                                  {airline, origin, destination, disruption_type,
                                              cause, jurisdiction, international} - all derived
                                              from the spec and from production's own routing,
                                              never hand-authored
    tags                                      a flat list: route, disruption, cause, carrier,
                                              edge kinds, difficulty, retrieval stress
    gold_docs, gold_sections, gold_doc_sections   the retrieval ground truth
    required_premises                         the premises an answer cannot be right without
                                              (golden/required.py): per premise, the sections that
                                              prove it ({any_of}) and which of those are the
                                              binding regulation ({primary}); [] on non-answer
                                              records
    required_doc_sections                     every any_of section, flattened, for reporting
    context                                   gold passages: {doc_id, doc_title, doc_type,
                                              source_class, section_id, breadcrumb, text};
                                              chunk-size independent (kept for
                                              evals/retrieval_eval.py's evidence metrics and
                                              generation_eval.py's reference contexts)
    expected_behavior.action                  "answer" | "clarify" | "abstain"
    expected_clarification                    the clarifying question - on "clarify" records
                                              only, null on the other two
    expected_facts                            key facts a correct answer must contain
    expected_output                           a short (usually <150 word) reference answer -
                                              an example, not something to match verbatim
    forbidden_claims                          materially wrong or unsafe claims to check for
    flight_fixture                            hybrid cases only (synthetic record)
    corpus_gap                                true if the entitlement relies on a source
                                              (e.g. CJEU case law) not present in data/
    category                                  spec.CATEGORIES taxonomy (kept for reporting)
    reply, assumption, unsupported_airline, notice   production's own output, kept so
                                              evals/generation_eval.py can replay
                                              a conversation and score the final turn. On an
                                              "answer" record a non-null `assumption` means
                                              production asks today where the golden set says it
                                              need not - a routing gap to fix, not a record shape
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "evals"):
    sys.path.insert(0, str(p))

OUT = ROOT / "evals" / "golden_set.jsonl"
MAX_WORDS = 150

_ROUTE_TAG_RE = re.compile(r"(US|UK|EU|OTHER)_to_(US|UK|EU|OTHER)")
ROUTE_TAGS = {
    "US_to_US", "UK_to_US", "EU_to_US", "UK_to_EU", "EU_to_UK", "US_to_UK", "US_to_EU",
    "OTHER_to_US", "OTHER_to_EU", "OTHER_to_UK", "US_to_OTHER", "EU_to_OTHER", "UK_to_OTHER",
    "US_EU_round_trip",   # two legs, one in each direction
    "unknown_direction",  # direction cannot be settled from the text -> ask, then assume
    "unspecified",        # no route given: no regime governs (US DOT retrievable), or,
                          # if production asks and gets no route, it assumes a US domestic flight
    "not_applicable",     # general or out-of-scope question
}

# Route-tag side -> production region name. The same names region_of() returns;
# "OTHER" (outside US/EU/UK) is never a stand-in for the US.
_TAG_REGION = {"US": "US", "EU": "EU", "UK": "UK", "OTHER": "OTHER"}


def expected_governing(tag: str, assumption: str | None = None) -> tuple[str, ...] | None:
    """The regimes the route tag implies, from the production matrix.

    `assumption` is the sentence production opened its answer with. It matters
    only for "unspecified": with no route at all nothing governs, unless
    production asked, got no route, and assumed a US domestic flight.
    """
    from flight_delay.jurisdiction import ASSUME_DOMESTIC, resolve_regimes

    m = _ROUTE_TAG_RE.fullmatch(tag)
    if m:
        return resolve_regimes(_TAG_REGION[m.group(1)], _TAG_REGION[m.group(2)])[1]
    if tag == "US_EU_round_trip":
        legs = resolve_regimes("US", "EU")[1] + resolve_regimes("EU", "US")[1]
        return tuple(j for j in ("US", "EU", "UK") if j in legs)
    if tag == "unspecified":
        if assumption == ASSUME_DOMESTIC:
            return resolve_regimes("US", "US")[1]
        return resolve_regimes(None, None)[1]
    return None


def check_route(spec):
    """
    Replay the case through production routing and compare with its route tag.
    Returns (problems, first_plan, final_plan, reply_used).
    """
    from golden.replay import default_reply, replay
    from golden.spec import CARRIERS

    from flight_delay.jurisdiction import region_of
    from flight_delay.tools import SUPPORTED_AIRLINES

    problems = []
    tag = spec["route"]

    m = _ROUTE_TAG_RE.fullmatch(tag)
    for code, idx, label in ((spec["origin"], 1, "origin"), (spec["dest"], 2, "dest")):
        if code and m and region_of(code) != _TAG_REGION[m.group(idx)]:
            problems.append(f"{label} {code} is {region_of(code)} in airport_regions.json, "
                            f"but route is {tag}")

    reply = spec.get("reply") or default_reply(spec["origin"], spec["dest"])
    first, final, used = replay(spec["question"], reply, spec["fixture"])

    if final.clarification is not None:
        problems.append("production asks a second clarifying question after the reply")
    if spec.get("reply") and first.clarification is None:
        problems.append("the case scripts a reply, but production answers without asking")

    carrier = CARRIERS.get(spec.get("carrier", "Unspecified"), "")
    if carrier:
        if final.flt.get("airline_scope") != carrier:
            problems.append(f"carrier {carrier}, but production scopes airline retrieval to "
                            f"{final.flt.get('airline_scope')!r}")
        unsupported = carrier not in SUPPORTED_AIRLINES
        if unsupported != (final.unsupported_airline == carrier):
            problems.append(f"carrier {carrier} is {'not ' if unsupported else ''}supported, but production "
                            f"reads unsupported_airline={final.unsupported_airline!r}")
        if unsupported and (first.live is not None or final.live is not None):
            problems.append("unsupported airline, but production used flight data for it")
    if tag == "unknown_direction" and first.clarification is None:
        problems.append(f"route is {tag}, but production answers without asking "
                        f"(reads governing={tuple(final.route.governing)})")
    if spec["fixture"] and first.clarification is not None:
        problems.append("live-data case, but production asks instead of using the flight record")

    want = expected_governing(tag, final.assumption)
    got = tuple(final.flt["jurisdiction_governing"])
    if want is not None and got != want:
        how = f"after the reply {used!r}" if used else "from the question"
        problems.append(f"route {tag} implies governing={want}, production reads {got} {how}")
    return problems, first, final, used


def load_sections():
    """{doc_id: [Section, ...]} for the whole manifest. No chunking, no tokenizer."""
    import index_corpus

    with contextlib.redirect_stdout(io.StringIO()):
        parsed, problems = index_corpus.parse_corpus()
    if problems:
        raise SystemExit(f"corpus did not parse cleanly: {problems}")
    return parsed


def _norm(s: str) -> str:
    return " ".join(s.split())


# The most words a `has=` passage grows to around its phrase (whole clause units
# only; the units containing the phrase are always kept, whatever their length).
GOLD_PASSAGE_WORDS = 100


@dataclass(frozen=True)
class Passage:
    section: object          # ingest Section
    first: int               # clause-unit index range [first, last]
    last: int

    def text(self, units) -> str:
        return "\n".join(units[self.first:self.last + 1])


def _units_around(units: list[str], needle: str) -> tuple[int, int] | None:
    """Unit range covering `needle`, grown to GOLD_PASSAGE_WORDS by whole neighbours."""
    starts, pos = [], 0
    for u in units:
        starts.append(pos)
        pos += len(_norm(u)) + 1
    joined = " ".join(_norm(u) for u in units)
    at = joined.find(needle)
    if at < 0:
        return None
    covered = [i for i, st in enumerate(starts)
               if st < at + len(needle) and st + len(_norm(units[i])) > at]
    first, last = covered[0], covered[-1]
    words = sum(len(units[i].split()) for i in range(first, last + 1))
    grow_after, grow_before = True, True
    while grow_after or grow_before:
        if grow_after:
            nxt = last + 1
            if nxt < len(units) and words + len(units[nxt].split()) <= GOLD_PASSAGE_WORDS:
                last, words = nxt, words + len(units[nxt].split())
            else:
                grow_after = False
        if grow_before:
            prv = first - 1
            if prv >= 0 and words + len(units[prv].split()) <= GOLD_PASSAGE_WORDS:
                first, words = prv, words + len(units[prv].split())
            else:
                grow_before = False
    return first, last


def resolve(ref, by_doc, units_of) -> list[Passage]:
    """
    Gold passages for one selector (golden/refs.py):
      section only  -> every clause of each section whose id starts with `section`
      has=          -> the clause units containing the phrase, in the FIRST section
                       (of `section`, or of the whole document) that contains it,
                       grown by whole neighbouring units to GOLD_PASSAGE_WORDS
    """
    sections = by_doc.get(ref.doc)
    if sections is None:
        raise ValueError(f"unknown doc_id {ref.doc!r}")
    if ref.section:
        sections = [s for s in sections if s.section_id.startswith(ref.section)]
    out: list[Passage] = []
    if not ref.has:
        out = [Passage(s, 0, len(units_of(s)) - 1) for s in sections]
    else:
        needle = _norm(ref.has)
        for sec in sections:
            span = _units_around(units_of(sec), needle)
            if span:
                out = [Passage(sec, *span)]
                break
    if not out:
        raise ValueError(f"selector matched nothing: {ref}")
    return out


def merge_passages(passages: list[Passage]) -> list[Passage]:
    """Overlapping or adjacent passages of one section become one, in first-seen order."""
    merged: list[Passage] = []
    for p in passages:
        for i, m in enumerate(merged):
            if m.section is p.section and p.first <= m.last + 1 and m.first <= p.last + 1:
                merged[i] = Passage(m.section, min(m.first, p.first), max(m.last, p.last))
                break
        else:
            merged.append(p)
    # A merge can make a passage touch another one merged earlier; repeat to a fixpoint.
    return merged if len(merged) == len(passages) else merge_passages(merged)


def load_cases():
    """The canonical cases, in spec.CANONICAL_CASES order (retired_cases.py is never loaded)."""
    from golden import cases_europe, cases_other, cases_us
    from golden.spec import CANONICAL_CASES

    cases = [*cases_us.CASES, *cases_europe.CASES, *cases_other.CASES]
    order = {cid: i for i, cid in enumerate(CANONICAL_CASES)}
    return sorted(cases, key=lambda c: order.get(c["id"], len(order)))


def canonical_errors(cases) -> list[str]:
    """Exactly the GOLDEN_SIZE ids in spec.CANONICAL_CASES, each defined once."""
    from golden.spec import CANONICAL_CASES, DIFFICULTIES, GOLDEN_SIZE, STRESS_KINDS

    errors = []
    ids = [c["id"] for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        errors.append(f"case ids defined more than once: {dupes}")
    if extra := sorted(set(ids) - set(CANONICAL_CASES)):
        errors.append(f"cases not in spec.CANONICAL_CASES (retire them or swap them in): {extra}")
    if missing := [i for i in CANONICAL_CASES if i not in ids]:
        errors.append(f"canonical cases with no spec: {missing}")
    if len(CANONICAL_CASES) != GOLDEN_SIZE or len(set(ids)) != GOLDEN_SIZE:
        errors.append(f"the golden set must have exactly {GOLDEN_SIZE} unique cases "
                      f"(canonical list {len(CANONICAL_CASES)}, case files {len(set(ids))})")
    for cid, (difficulty, stress) in CANONICAL_CASES.items():
        if difficulty not in DIFFICULTIES or not set(stress) <= STRESS_KINDS:
            errors.append(f"{cid}: difficulty/stress {difficulty!r}/{stress!r} not in vocabulary")
    return errors


_CONTROL_SNAKE = {
    "Controllable": "controllable",
    "Uncontrollable": "uncontrollable",
    "Unknown": "unknown_cause",
    "Passenger-caused": "passenger_caused",
    "N/A": "na",
}

# Keyword -> scenario.cause. Order matters (checked first match wins); these
# are a heuristic reading of prose the spec already wrote, not a new fact the
# case author has to state twice.
_CAUSE_KEYWORDS = (
    ("snow", "weather"), ("storm", "weather"), ("weather", "weather"), ("fog", "weather"),
    ("hurricane", "weather"), ("ice", "weather"),
    ("mechanical", "mechanical"), ("maintenance", "mechanical"), ("technical", "mechanical"),
    ("crew shortage", "staffing"), ("crew scheduling", "staffing"), ("shortage of labor", "staffing"),
    ("air traffic control", "air_traffic_control"),
    ("smaller plane", "operational"), ("smaller-capacity", "operational"),
    ("smaller aircraft", "operational"), ("operational reasons", "operational"),
    ("strike", "strike"),
    ("oversold", "overbooking"), ("oversale", "overbooking"), ("bumped", "overbooking"),
    ("denied boarding", "overbooking"),
)


def _snake(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_")


def _infer_cause(spec: dict) -> str:
    """A short cause word for scenario.cause. Passenger-caused/N/A come straight
    from `control`; otherwise a keyword scan of the question and drafted answer,
    since the spec already states the cause in prose and shouldn't state it twice."""
    control = spec["control"]
    if control == "Passenger-caused":
        return "passenger_caused"
    if control == "N/A":
        return "na"
    text = f"{spec['question']} {spec['expected']}".lower()
    for kw, cause in _CAUSE_KEYWORDS:
        if kw in text:
            return cause
    return "unknown"


def _international(route: str, assumption: str | None) -> bool | None:
    """None when the route is genuinely unresolved (an unrecognised city with no
    country, or a direction question with no reply that settles it)."""
    from flight_delay.jurisdiction import ASSUME_DOMESTIC

    if route == "US_to_US":
        return False
    if route in ("unspecified", "not_applicable"):
        return False if assumption == ASSUME_DOMESTIC else None
    if route == "unknown_direction":
        return None
    return True


def build(cases, sections_by_doc):
    from golden.refs import BINDING_DOCS
    from golden.required import REQUIRED
    from golden.spec import (
        CANONICAL_CASES,
        CARRIERS,
        CATEGORIES,
        CONTROLLABILITY,
        DISRUPTIONS,
        EDGE_DERIVED,
        EDGE_KINDS,
    )

    from flight_delay.ingest import clause_units

    unit_cache: dict[int, list[str]] = {}

    def units_of(section):
        if id(section) not in unit_cache:
            unit_cache[id(section)] = clause_units(section.text)
        return unit_cache[id(section)]

    errors: list[str] = canonical_errors(cases)
    records = []
    seen_ids, seen_q = set(), set()

    for spec in cases:
        cid = spec["id"]
        err = errors.append

        if cid in seen_ids:
            err(f"{cid}: duplicate id")
        seen_ids.add(cid)
        if spec["question"].lower() in seen_q:
            err(f"{cid}: duplicate question")
        seen_q.add(spec["question"].lower())

        for field, vocab in (("carrier", CARRIERS), ("disruption", DISRUPTIONS),
                             ("control", CONTROLLABILITY), ("category", CATEGORIES),
                             ("route", ROUTE_TAGS)):
            if spec[field] not in vocab:
                err(f"{cid}: {field}={spec[field]!r} not in vocabulary")

        problems, first, final, reply_used = check_route(spec)
        for problem in problems:
            err(f"{cid}: {problem}")
        asked = first.clarification is not None
        declined = first.unsupported_reply is not None

        clarify = spec["clarify"]
        if spec["route"] == "unknown_direction" and not clarify:
            err(f"{cid}: route unknown_direction is a case production must stop and ask on; "
                f"set clarify=True (and answerable=False)")
        if clarify and spec["answerable"]:
            err(f"{cid}: a clarify case has no answer to score yet; set answerable=False")

        # expected_output is production's own text for a decline or an
        # unresolved clarification; the spec's drafted answer otherwise.
        #
        # Exactly one of three shapes per record, so a reader never has to
        # reconcile two of them (the reason `clarify` is checked before
        # `answerable`, and why the assumption sentence is no longer glued onto
        # an "answer" record's reference text):
        #   answer   answerable=True,  expected_clarification=None
        #   clarify  answerable=False, expected_clarification=the question asked
        #   abstain  answerable=False, expected_clarification=None
        if declined:
            if spec["expected"]:
                err(f"{cid}: production declines (unsupported airline); leave `expected` empty")
            if spec["answerable"] or spec["refs"]:
                err(f"{cid}: a declined case is a trap with no refs")
            expected, action = first.unsupported_reply, "abstain"
        elif clarify:
            if spec["expected"]:
                err(f"{cid}: a clarify case expects production's own clarifying question as "
                    f"expected_output; leave `expected` empty and use expected_facts for what a "
                    f"correct final answer must get right")
            if not asked:
                err(f"{cid}: clarify case, but production answers without asking")
            expected, action = first.clarification or "", "clarify"
        elif not spec["answerable"]:
            if not spec["expected"]:
                err(f"{cid}: empty `expected`, but the case is an unanswerable trap")
            expected, action = spec["expected"], "abstain"
        else:
            if not spec["expected"]:
                err(f"{cid}: empty `expected`, but production answers")
            # No "Assuming you're flying from ..." sentence here: this record
            # says the question can be answered as asked. Production's actual
            # assumption stays in the `assumption` field for replay scoring.
            expected = "\n\n".join(t for t in (final.notice, spec["expected"]) if t)
            action = "answer"
        if action == "answer":
            words = len(expected.split())
            if words > MAX_WORDS:
                err(f"{cid}: expected_output is {words} words (> {MAX_WORDS})")

        if spec["edge"] is not None and spec["edge"] not in EDGE_KINDS - EDGE_DERIVED:
            err(f"{cid}: edge={spec['edge']!r} is derived or not in vocabulary")
        edge = [kind for kind, hit in (
            ("ambiguous_route", asked),
            ("out_of_scope", spec["category"] == "trap"),
            ("passenger_caused", spec["control"] == "Passenger-caused"),
            ("unsupported_airline", final.unsupported_airline is not None),
        ) if hit] + ([spec["edge"]] if spec["edge"] else [])

        if spec["answerable"] and not spec["refs"]:
            err(f"{cid}: answerable case with no context refs")
        if not spec["answerable"] and not clarify and spec["category"] != "trap":
            err(f"{cid}: unanswerable cases must use category 'trap' (or be clarify cases)")
        if not spec["expected_facts"]:
            err(f"{cid}: no expected_facts")
        if not spec["forbidden_claims"]:
            err(f"{cid}: no forbidden_claims")

        passages: list[Passage] = []
        for ref in spec["refs"]:
            try:
                passages += resolve(ref, sections_by_doc, units_of)
            except ValueError as e:
                err(f"{cid}: {e}")
        context = []
        for p in merge_passages(passages):
            sec = p.section
            context.append({
                "doc_id": sec.doc_id,
                "doc_title": sec.doc_title,
                "doc_type": sec.doc_type,
                "source_class": sec.source_class,
                "section_id": sec.section_id,
                "breadcrumb": sec.breadcrumb,
                "text": p.text(units_of(sec)),
            })

        # Required premises (golden/required.py): every answer case has some, and
        # nothing else does. A premise lists the sources that PROVE it, any one of
        # which counts; at least one of them must be among the case's own gold refs,
        # so a premise stays anchored in the reviewed evidence, while an equivalent
        # statement of the same rule (the DOT Q&A for Part 260, the CAA for UK261)
        # may be an alternate that is not itself labelled gold.
        premises = REQUIRED.get(cid, [])
        if action == "answer" and not premises:
            err(f"{cid}: answer case with no required premises (golden/required.py)")
        if action != "answer" and premises:
            err(f"{cid}: only answer cases carry required premises")
        required_premises: list[dict] = []
        for group in premises:
            if not any(ref in spec["refs"] for ref in group):
                err(f"{cid}: no source of a required premise is one of the case's refs: {group}")
                continue
            any_of, primary = [], []
            for ref in group:
                with contextlib.suppress(ValueError):   # already reported above
                    for p in resolve(ref, sections_by_doc, units_of):
                        key = f"{p.section.doc_id}#{p.section.section_id}"
                        any_of.append(key)
                        # The binding text, as opposed to a regulator's guidance on it
                        # or a carrier's own promise: primary_authority_coverage.
                        if p.section.doc_id in BINDING_DOCS:
                            primary.append(key)
            required_premises.append({"any_of": list(dict.fromkeys(any_of)),
                                      "primary": list(dict.fromkeys(primary))})
        required_sections = [s for g in required_premises for s in g["any_of"]]

        if final.unsupported_airline and any(c["source_class"] == "airline" for c in context):
            err(f"{cid}: unsupported airline, but the gold context uses airline documents")

        difficulty, stress = CANONICAL_CASES.get(cid, (None, ()))
        # For a "clarify" case, scenario/tags describe what's known BEFORE the
        # scripted reply - the whole point is that it isn't resolved yet - not
        # whatever the one-round default-reply fallback happens to settle on.
        route_state = first if action == "clarify" else final
        governing = [] if spec["route"] == "not_applicable" else list(route_state.flt["jurisdiction_governing"])
        tags = list(dict.fromkeys([
            spec["route"].lower(),
            _snake(spec["disruption"]),
            _CONTROL_SNAKE.get(spec["control"], _snake(spec["control"])),
            _snake(spec["carrier"]),
            *edge,
            *([difficulty] if difficulty else []),
            *stress,
        ]))
        rec = {
            "id": cid,
            "question": spec["question"],
            "answerable": spec["answerable"],
            "scenario": {
                "airline": CARRIERS.get(spec["carrier"], ""),
                "origin": spec["origin"],
                "destination": spec["dest"],
                "disruption_type": _snake(spec["disruption"]),
                "cause": _infer_cause(spec),
                "jurisdiction": "+".join(governing) or "none",
                "international": _international(spec["route"], final.assumption),
            },
            "tags": tags,
            "gold_docs": list(dict.fromkeys(c["doc_id"] for c in context)),
            "gold_sections": list(dict.fromkeys(c["section_id"] for c in context)),
            "gold_doc_sections": list(dict.fromkeys(f"{c['doc_id']}#{c['section_id']}" for c in context)),
            "required_premises": required_premises,
            # Flattened, for reporting only: every source that could prove a premise.
            "required_doc_sections": list(dict.fromkeys(required_sections)),
            "expected_behavior": {"action": action},
            # Only a "clarify" record carries one: an "answer" record asserts
            # the question is answerable as asked, and an "abstain" record has
            # nothing to clarify. Production's question, when it asks anyway on
            # an "answer" case, is still reachable via check_route/replay.
            "expected_clarification": first.clarification if action == "clarify" else None,
            "expected_facts": spec["expected_facts"],
            "expected_output": expected,
            "forbidden_claims": spec["forbidden_claims"],
            "flight_fixture": spec["fixture"],
            "corpus_gap": bool(spec["gap"]),
            # Kept beyond the requested schema because evals/retrieval_eval.py
            # and evals/generation_eval.py depend on them (CLAUDE.md: retrieval
            # evidence-text metrics, and replaying a clarify-then-answer
            # conversation to score the final turn) - see module docstring.
            "category": spec["category"],
            "context": context,
            "reply": reply_used,
            "assumption": final.assumption,
            "unsupported_airline": final.unsupported_airline,
            "notice": final.notice,
        }
        records.append(rec)

    errors += edge_share_errors(records)
    return records, errors


def edge_share_errors(records) -> list[str]:
    """At least EDGE_MIN_SHARE edge/ambiguous cases, overall and among answerable cases."""
    from golden.spec import EDGE_KINDS, EDGE_MIN_SHARE

    def has_edge(r):
        return any(t in EDGE_KINDS for t in r["tags"])

    errors = []
    for label, subset in (("all", records), ("answerable", [r for r in records if r["answerable"]])):
        if not subset:
            continue
        share = sum(has_edge(r) for r in subset) / len(subset)
        if share < EDGE_MIN_SHARE:
            errors.append(f"edge/ambiguous cases are {share:.0%} of {label} cases "
                          f"(minimum {EDGE_MIN_SHARE:.0%}): add or tag edge cases")
    return errors


def report(records):
    from golden.spec import DIFFICULTIES, EDGE_KINDS, STRESS_KINDS

    def show(title, counter):
        print(f"  {title}: " + ", ".join(f"{k} {v}" for k, v in counter.most_common()))

    def has_edge(r):
        return any(t in EDGE_KINDS for t in r["tags"])

    # "not answerable" now covers two different things: a trap/decline with
    # nothing in the corpus to answer from, and a clarify case that cannot be
    # answered until the passenger fills in the missing detail.
    actions = Counter(r["expected_behavior"]["action"] for r in records)
    print(f"\n{len(records)} cases, {len({r['id'] for r in records})} unique ids "
          f"({sum(r['answerable'] for r in records)} answerable, "
          f"{actions['clarify']} clarify, {actions['abstain']} abstain)")
    show("category", Counter(r["category"] for r in records))
    show("action", Counter(r["expected_behavior"]["action"] for r in records))
    show("airline", Counter(r["scenario"]["airline"] or "unspecified" for r in records))
    show("disruption", Counter(r["scenario"]["disruption_type"] for r in records))
    show("cause", Counter(r["scenario"]["cause"] for r in records))
    show("jurisdiction", Counter(r["scenario"]["jurisdiction"] for r in records))
    show("difficulty", Counter(t for r in records for t in r["tags"] if t in DIFFICULTIES))
    show("retrieval stress", Counter(t for r in records for t in r["tags"] if t in STRESS_KINDS))
    show("source mix", Counter("+".join(sorted({c["source_class"] for c in r["context"]})) or "none"
                               for r in records))
    for label, subset in (("edge/ambiguous, all", records),
                          ("edge/ambiguous, answerable", [r for r in records if r["answerable"]])):
        n = sum(has_edge(r) for r in subset)
        print(f"  {label}: {n}/{len(subset)} ({n / len(subset):.0%})")
    show("edge kinds", Counter(t for r in records for t in r["tags"] if t in EDGE_KINDS))
    print(f"  corpus gaps flagged: {sum(r['corpus_gap'] for r in records)}")
    words = [len(r["expected_output"].split()) for r in records]
    print(f"  expected_output words: max {max(words)}, mean {sum(words) // len(words)}")
    docs = Counter(d for r in records for d in r["gold_docs"])
    print(f"  documents used as gold: {len(docs)} -> " + ", ".join(f"{d} {n}" for d, n in docs.most_common()))


def saved_file_drift(records, path=None) -> list[str]:
    """
    Differences between freshly built records and the saved JSONL, compared as
    parsed JSON (so escaping and key order do not count). Empty when they agree.

    Validating a fresh build proves the SPECS are consistent; it says nothing about
    the file both evaluation stages actually read, which can be stale.
    """
    path = OUT if path is None else path
    if not path.exists():
        return [f"{path.name} does not exist"]
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    problems = []
    built_ids, saved_ids = [r["id"] for r in records], [r["id"] for r in saved]
    if built_ids != saved_ids:
        missing = [i for i in built_ids if i not in saved_ids]
        extra = [i for i in saved_ids if i not in built_ids]
        problems.append(f"case ids/order differ (missing {missing}, extra {extra})")
    by_id = {r["id"]: r for r in saved}
    for r in records:
        old = by_id.get(r["id"])
        if old is None:
            continue
        fresh = json.loads(json.dumps(r))
        fields = sorted(k for k in set(fresh) | set(old) if fresh.get(k) != old.get(k))
        if fields:
            problems.append(f"{r['id']}: {', '.join(fields)}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate, and compare with the saved JSONL; never writes")
    args = ap.parse_args()

    records, errors = build(load_cases(), load_sections())

    if errors:
        print("BUILD FAILED:")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)

    report(records)
    if args.check:
        drift = saved_file_drift(records)
        if drift:
            print(f"\n--check: {OUT.name} is STALE against the case specs ({len(drift)} differences):")
            for d in drift:
                print("  -", d)
            print("Rebuild it with evals/build_golden_set.py (nothing was written).")
            raise SystemExit(1)
        print(f"\n--check: {OUT.name} matches a fresh build. Nothing written.")
        return
    # ensure_ascii=True is deliberate: anything opening this file with the
    # platform default encoding (cp1252 on Windows) cannot decode the €, £ and §
    # in the answers and chunk text. Escaped JSON loads anywhere.
    with OUT.open("w", encoding="ascii") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
