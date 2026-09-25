# Results

Evaluation runs in two stages. Stage 1 selects a chunk configuration; Stage 2
generates answers at that configuration and judges them.

| stage | command | writes |
|---|---|---|
| 1. retrieval — select a chunk configuration | `evals/retrieval_eval.py` | `evals/runs/retrieval/<stamp>.json` + `latest.json` |
| 2. generation — Ragas judge + safety metrics | `evals/generation_eval.py` | `evals/runs/generation/<stamp>.json` + `latest.json` |

See `RUNBOOK.md` for the exact commands (`make` cannot find `python` on this
machine). Stage 2 makes **paid** API calls and needs explicit permission for
every run.

---

## Stage 1 — measured, authoritative

Real embedder (`bge-large-en-v1.5`), real reranker (`bge-reranker-base`), on
PostgreSQL + pgvector, all 43 scored cases. Source of truth:
`evals/runs/retrieval/latest.json` (= `20260920T000831.json`, Session 25). Quote this
file, not a superseded run.

**Selected: 512 tokens / 15% overlap**, with the Session 25 retrieval configuration
("G"): scope lanes over every regime in `jurisdiction_scope`, conditional US
procedural lanes, a reserved SEAT for each lane's best hit, and a reduced airline
quota for questions no carrier page can answer. `final_k` stays 8.

| metric | **G — adopted** | B (Session 24) | A (Session 22) |
|---|---|---|---|
| `required_premise_complete` | **0.744** (32/43) | 0.372 (16/43) | 0.302 (13/43) |
| `primary_authority_coverage` | **0.703** | 0.370 | 0.313 |
| `evidence_recall` | **0.562** | 0.514 | 0.474 |
| `evidence_hit_rate` | **0.550** | 0.487 | — |
| `context_precision` | **0.218** | 0.167 | — |
| `section_recall@3` | 0.265 | 0.264 | 0.232 |
| `MRR` | 0.647 | 0.660 | 0.639 |
| `context_tokens` | 3,671 | 3,570 | 3,539 |
| `duplicate_section_chunks` | 46 | 56 | — |
| `governing_kept_share` | 1.000 | 1.000 | 1.000 |

MRR and nDCG@5 rank the whole candidate pool BEFORE balancing, so lane votes move
them; the context the application actually consumes got both more complete and more
precise over the same change. `required_premise_complete` is the deciding metric.

The three premise/evidence numbers answer three different questions (Session 24,
`evals/golden/required.py`, reviewed by the user): was every premise proved by SOME
accepted source; was the binding regulation itself present rather than only a
regulator's guidance repeating it; and how much of all the useful gold text was
retrieved. Equivalent sources count for the first (DOT Q&A / Part 260, CAA / UK261,
Commission guidelines / EU261). Scope premises are required because prompt Rule 3
makes "UK261 applies" a claim that must cite the scope text.

**11 cases still miss a premise** (from 27 at B): uk-us-03, uk-us-09, uk-us-14,
eu-us-07, eu-us-13, us-eur-09, intra-04, intra-07, law-03, law-07, live-05. Six of
them get no remedy lane at all, two seat guidance where the binding article is
required, and live-05 is a corpus gap. See `previous_trials.md`.

Every configuration that was measured and NOT adopted — chunk size 384, `final_k` 9,
section-diverse lane seats, the section cap, the Session 22 knob ablations — is
recorded with its numbers and its reason in **`previous_trials.md`**.

`governing_kept_share` is the metric that earns its place. It read 0.9268 with
three named cases (uk-us-09, eu-us-12, law-03) while nobody treated it as a
defect (Session 19). It is 1.000 through Sessions 22-25, but note what it cannot
see: with the CAA guidance filed as `regulation` it also read 1.000 while live-01
kept no UK261 article at all, because a CAA page satisfied "UK law present". Guidance
now never fills a regime's seat.

## Stage 2 — provisional, NOT authoritative

The authoritative 50-case run **has never completed**. What exists:

- A free routing + gate-calibration pass (`--generator none`).
- A **10-case stratified subset** run that provisionally selected
  `groq-gpt-oss-20b` (`reasoning_effort: low`, cap 900). `.env` carries it so
  `/ask` starts. It is recorded as provisional everywhere.
- A first full attempt that died on quota: only 11 of 50 cases ever reached the
  model, all `dom-*`, so every generation metric in that run describes the US
  domestic slice. Its `abstention_accuracy FAILED trap-04` is an artifact —
  trap-04 never got a response.

Why it has not finished: one call costs ~5,700 tokens against a **200,000
token/day** free cap, so a 50-case run needs ~285k without retries and ~493k at
the observed retry rate — 1.4x to 2.5x the daily cap. It needs ~2.5 days of
quota, or a resumable generation pass with an on-disk answer cache, which does
not exist yet.

### The basis of every generation number: 10 sampled questions, not 50

**All generation metrics below were measured on 10 golden-set questions sampled
from the 50, because the free-tier quota does not stretch to a full run. No
full 50-case run has ever been executed.** Source of record:
`evals/runs/generation/20260920T014936.json` (Session 26, the first Stage 2 run on
the G retrieval configuration; same 10 cases, seed 3151573671, so it compares
directly with `20260919T222116.json` on the old retrieval).

### Retrieval improved; generation did not (Session 26)

| | OLD retrieval (B) | NEW retrieval (G) |
|---|---|---|
| answers delivered | 5 of 8 | **4 of 8** |
| validation failures | 3 | **4** |
| factual_correctness (all 8) | 0.191 | **0.089** |
| factual_correctness (judged only) | 0.306 | **0.177** |
| faithfulness | 0.703 (n=5) | 0.667 (n=4) |
| citation_validity | 1.000 | 1.000 |
| tokens/case, cost | 11,211, $0.0074 | 11,902, $0.0079 |

Net **one case**: dom-24 and uk-us-01 went answered → validation_failed, law-07 went
the other way. Both regressions are a **total citation collapse** (3→0 and 4→0 `[Sn]`
markers) on cases whose *retrieval* G had fixed — the evidence was present and the
model stopped citing it. law-07 got 47% more context and started citing correctly,
so this is not monotonic in context size.

**The bottleneck has moved from retrieval to citation discipline.** Half the answer
cases deliver nothing and every failure is the same rule: a factual sentence with no
`[Sn]`. `validate_answer` is behaving correctly; the 20B model is not meeting the
bar and one retry does not rescue it. The retrieval configuration was NOT reverted:
a deterministic 16→32 of 43 gain across the whole scored set outweighs one case of
eight on a subset the harness labels non-authoritative.

| field | value |
|---|---|
| `n_cases` | **10** of 50 |
| `sample` | `{n: 10, seed: 3151573671, strategy: "stratified by category", of: 50}` |
| `authoritative` | **`false`** |
| `full_run` | **`false`** |
| generator | `groq-gpt-oss-20b`, `reasoning_effort: low`, `max_completion_tokens` 900 |
| gate | `--gate off` |

The draw is **stratified by `category`, not uniform**: a uniform draw of 10
omits the clarify or abstain cases about half the time, and those are the hard
gates — an empty gate passes vacuously and reads as a success. The seed is
recorded so the draw repeats.

Measured on those 10 cases (8 reached the model; 2 are routing-only), beside the
same run before Sessions 22-23 changed the corpus, the prompt and retrieval:

| metric | Session 24 | n | before (Session 22 run) | read it as |
|---|---|---|---|---|
| `faithfulness` | 0.703 | **5** | 0.724 (n=4) | to authorized evidence; one more answer is included now |
| `factual_correctness` | **0.191** | 8 | 0.128 | 3 of 8 delivered no answer -> scored 0.0 (was 4) |
| `factual_correctness_judged_only` | **0.306** | 5 | 0.255 | the answers actually delivered |
| `validation_pass_rate` | **0.625** | 8 | 0.50 | now failing: eu-us-01, law-07, live-01 (uk-us-09 and intra-07 fixed) |
| `first_try_pass_rate` | 0.25 | 8 | 0.375 | a retry is still the norm |
| `citation_validity` | 1.000 | 8 | 1.000 | no fabricated or unknown citation markers |
| `llm_error_rate` | 0.000 | 8 | 0.000 | |
| `abstention_accuracy` | 1.000 | **1** | 1.000 | one case — not a result |
| `clarification_accuracy` | 1.000 | **1** | 1.000 | one case — not a result |
| `unsupported_airline_accuracy` | 1.000 | **1** | 1.000 | one case — not a result |
| `scope_safety` | 1.000 | **1** | 1.000 | one case — not a result |

Cost: $0.0074 for the 10 cases ($0.00093/case, 11,211 tokens/case); the judge's
cost is unavailable by design (`JUDGE_PRICES = None`). Extrapolated generator cost
for a full 44-case generation pass: about $0.04.

The four safety gates each rest on a **single case**, and faithfulness on four.
The 1.000s are not evidence that the gates hold; they are evidence that one case
each passed. Quote the `n` beside every number or the table misleads.

A separate live finding, Session 21: the served app produced an answer that
misstated three UK261 thresholds while citing sources correctly, from evidence
that was complete and unsplit in its context. `citation_validity` 1.000 does not
mean the cited clause supports the number — the validator checks that a claim
*has* a citation, not that the citation bears it.

The judge is **Google AI Studio `gemini-3.5-flash-lite`**, frozen, never a
candidate. `JUDGE_PRICES` is `None`, so judge cost reports as *unavailable*
rather than as a fabricated zero.

**Do not quote a Stage 2 number as a result of this system** until the full run
completes. A subset run may select a generator, but it carries `full_run: false`
and a `basis` string, and both are meant to be read.

## Carried into any result

The golden set's expected answers are **unvetted drafts**, so factual
correctness against them is a consistency signal, not validated accuracy. The
confidence gate is uncalibrated and currently harmful on the golden set (it
refused 19 of 41 answer cases that reached it and let the only trap through), so
every measured run uses `--gate off` and the served app sets
`CONFIDENCE_GATE_ENABLED=false`.

No retrieval-quality metric is exported at runtime, by design: the application
does not compute recall, and an alert on a metric nothing emits never fires.
