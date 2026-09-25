# Previous trials — retrieval configurations that were measured and not adopted

The record of what was tried, what it scored and why it was dropped, so nothing here is
retried blindly. **The adopted configuration is G** (`config.py` defaults,
`evals/runs/retrieval/20260920T000831.json` = `latest.json`). Full narrative is in
PROGRESS.md; this file is the short version.

All numbers: Stage 1, 512 tokens / 15% overlap, 43 scored golden cases, PostgreSQL +
pgvector, real `bge-large-en-v1.5` + `bge-reranker-base`. No LLM is called in Stage 1.
`required_premise_complete` is the deciding metric (`evals/golden/required.py`).

## The lineage

| | what changed | premise-complete | primary authority | evidence recall | verdict |
|---|---|---|---|---|---|
| A | Session 22 code | 0.302 (13/43) | 0.313 | 0.474 | superseded |
| B | + Session 23 remedy lanes, tarmac/complaint wording, fusion-pool fix | 0.372 (16/43) | 0.370 | 0.514 | superseded |
| C | B at chunk size 384 | 0.372 (16/43) | 0.309 | 0.488 | **rejected** — ties on premises, loses primary authority |
| D | B + scope lanes + US procedural lanes | 0.465 (20/43) | 0.484 | 0.496 | kept, partial |
| E | D + airline quota 1 for law-only questions | 0.465 (20/43) | 0.492 | 0.499 | kept |
| F | E + seats for scope/procedural lane leaders | 0.651 (28/43) | 0.683 | 0.548 | kept |
| **G** | F + seats for remedy lane leaders | **0.744 (32/43)** | **0.703** | **0.562** | **ADOPTED** |
| H | G + `final_k` 9 | 0.744 (32/43) | 0.703 | 0.573 | **rejected** |
| I | G + section-diverse seats + a 2nd seat per remedy lane | 0.698 (30/43) | 0.695 | 0.566 | **rejected** |
| I-a | G + section-diverse seats only | 0.698 (30/43) | 0.683 | 0.561 | **rejected** |

B was re-run with every Session 25 knob off and reproduced A→B's numbers exactly, so the
columns are comparable rather than merely sequential.

## What did not work, and why

**C — chunk size 384.** Identical premise coverage on the SAME 16 cases; 512 wins on primary
authority (0.370 vs 0.309) and MRR. Re-index + one free Stage 1 run is all it costs, so 384
stays available if truncation ever shows up as misread clauses (it cuts reranker pair
truncation 46.1% → 26.5%). Do not re-run it while tuning retrieval.

**H — `final_k` 9.** The same 32 cases, the same primary authority, +350 context tokens,
duplicate-section chunks 46 → 56, and `kept_of_retrieved` 0.994 → 0.977 (the budget starts
dropping what was retrieved). The remaining failures are not a context-capacity problem.
`final_k` stays 8.

**I / I-a — a second, section-diverse lane seat.** It does what it says: duplicate-section
chunks 46 → 37. It still costs two cases. uk-us-06 and unsup-01 both lost `uk-261#article-8`
and gained a DOT Q&A page on rebooking offers. The cause is the fall-through rule, not the
second seat (I and I-a score identically): when a lane's best hit is already in the context,
G lets the lane LOSE its seat and the slot goes back to relevance, which is what put Article 8
there. With fall-through the lane spends that slot on its own second-best candidate.

> **A lane's second-best hit is worth less than the ranking's next pick.** A lane makes ONE
> offer; a seat it cannot use belongs back with relevance.

Knobs kept, defaulted off, with the measurement in their `config.py` comments:
`lane_seat_section_diverse`, `max_seats_per_lane`.

**Earlier, still valid:** `max_chunks_per_section` — a cap of 2 lowered evidence recall
0.474 → 0.459 (gold passages in long sections span chunks). Stays 0.

## Not tried, deliberately

**49 U.S.C. §40102 as a neutral US-territory source** (would fix live-05, whose only source is
Part 250's denied-boarding notice that the topic filter removes). It is a `data/` change and
`data/` is the user's to curate; it also moves `required.py` and the golden digest, which would
have made the table above incomparable. Waiting on the user's document.

**Weakening the topic filter to recover live-05.** Rejected: it would reopen Part 250 for every
unrelated cancellation question.

## Where the remaining 11 failures actually come from

Diagnosed by replaying each failing case through the production pipeline and printing its seats
(not guessed from the metric):

1. **Six have no remedy lane firing at all** — uk-us-09, uk-us-14, eu-us-13, us-eur-09,
   intra-07, law-07. `pipeline.remedy_lanes` needs `_OPEN_ENTITLEMENT` to match AND `governing`
   to be non-empty; "Can I get a refund under US DOT rules?" matches no remedy word, and law-07
   has an empty `governing` because it is a general comparative question. A trigger fix, not a
   ranking change — the highest-value next step.
2. **Two seat guidance where the premise needs the binding article** — intra-04 and law-03 seat
   the Commission guidelines where `required.py` asks for EU261 Art 7 / Art 8 / Art 15.
   `retrieval._binding` already protects the governing guarantee from this; lane seats have no
   equivalent.
3. **live-05** — a corpus gap (see above), not a ranking failure.
4. **law-03 / law-07** may still want their own intents (`legal_exception`, `comparative_law`)
   rather than another global ranking change.
