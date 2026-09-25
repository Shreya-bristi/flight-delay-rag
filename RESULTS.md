# Results

These are the numbers from my final evaluation runs, the metrics that are strong and the ones
that aren't, and what each metric means for this project.

## How I evaluate

The test set is a hand-built **golden set of 50 questions**. It covers:

- US domestic, EU/UK to US, US to Europe and intra-Europe routes
- live flight lookups
- questions that need a clarifying question first
- trick questions the bot should decline
- airlines it doesn't support

More than half the cases are edge cases on purpose: ambiguous routes, a regime that doesn't
apply, exception clauses, multi-leg trips, US territories.

Evaluation runs in two stages:

1. **Retrieval.** Try to answer if search put the right regulation and airline text in front of the
   model. This needs no LLM, runs on the real stack (PostgreSQL + pgvector, bge-large-en-v1.5,
   bge-reranker-base), and scores the **43 answerable questions**. The other 7 are questions
   where the right move is to ask or decline, so there is nothing to retrieve.
2. **Generation.** Here I try to answer if the answer is correct, grounded in the sources and properly cited, and the bot ask or decline when it should. The generator is `openai/gpt-oss-20b` on Groq.
   A separate judge, Gemini Flash-3.5 Lite, scores the answers, so a model never grades itself (tried to implement llm-as -a judge).

| run | file | scope |
|---|---|---|
| Retrieval (final) | `evals/runs/retrieval/latest.json` | all 43 scored questions, 512-token chunks, 15% overlap |
| Generation (final) | `evals/runs/generation/20260920T032529.json` | 10-question sample, stratified by category |

Generation is a sample because of Groq's free tier. One question costs about 5,700 tokens,
and the free limit is 200,000 tokens a day, which a 50-question run with retries doesn't fit
into. The sample is stratified so it always includes a clarify case, a decline case and an
unsupported-airline case; a random 10 would often miss them.

---

## Retrieval: what shines through

| metric | result | what it means here |
|---|---|---|
| **Required premise completeness** | **0.744** (32 of 43) | For each question I listed every fact the answer depends on (for example "UK261 applies to this flight", "delay of 3h+ triggers compensation", "the amount for this distance"). This is the share of questions where **every one** of those facts was in the retrieved context. It's the metric I optimised for, because a missing premise means the model has to guess or leave something out. |
| **Primary authority coverage** | **0.722** | The share of required legal facts backed by the **binding text itself** (the EU261 article, the 14 CFR section) rather than only by guidance that paraphrases it. It matters because a passenger arguing with an airline needs the regulation, not a summary. |
| **Governing regime kept** | **1.000** | The law that governs the flight (decided by the departure airport) gets a guaranteed slot in the context. This checks that the slot survives the token budget in every case. It once read 0.93, and that turned out to be a real bug: the guaranteed chunk was the first one dropped when the context got tight. |
| **Off-carrier chunks** | **0** | A United question never gets Delta's contract in its context. |
| **Cases with zero evidence** | **0** | Every answerable question retrieved at least some gold text. |
| **Dense search recall vs exact** | **1.000** | The approximate vector search (HNSW) returns the same top 20 as an exact scan.  |
| **Gold passage in one chunk** | **0.94** | At 512 tokens, 94% of the passages that answer a question fall inside a single chunk instead of being split across two. |

**How far it moved.** The same metrics on my earlier retrieval design, same golden set:

| metric | before | final |
|---|---|---|
| Required premise completeness | 0.302 (13 of 43) | **0.744** (32 of 43) |
| Primary authority coverage | 0.313 | **0.722** |
| Evidence recall | 0.474 | **0.571** |

Most of that came from **search lanes**. For a question like "what am I owed?", each regime
that applies gets extra targeted searches (compensation amount, care, refund, scope), and the
best hit from each lane is guaranteed a seat in the context. Without the seats, the right
article was often found and then lost in the final ranking.

Things I measured and didn't adopt, because they didn't help:

- 384-token chunks
- a 9th context slot
- a cap on chunks from the same section

one definition first. A gold section is a corpus section (like uk-261#article-7-right-to-compensation) that the golden set marks as containing part of a question's correct answer. Retrieval is scored on whether those sections, and the gold passages inside them, get found.

## Retrieval: what's weak

| metric | result | what it means here, and why it's low |
|---|---|---|
| **Evidence recall** | 0.571 | The share of **all** useful gold text that was retrieved, not just the required facts. Many questions have more relevant text than fits in 8 chunks, so this won't reach 1.0 by design. It's the completeness metric above that decides whether the answer can be right. |
| **Context precision** | 0.323 |  |
| **Section recall@3** | 0.362 | The share of gold sections (scored as document + section, since EU261 and UK261 share article numbers) that appear in the top 3. A question often needs 3 or more sections, so the top 3 can't hold them all. |
| **nDCG@5** | 0.364 | This measures how well the most relevant results are ranked in the top 5. It is calculated before the later source-balancing and lane-selection steps, so the final context shown to the model can be better than this score suggests. |
| **MRR** | 0.647 | Mean reciprocal rank of the first gold chunk, on the same pre-balancing ranking. The first useful chunk is usually near the top, but not always first. |

| **Reranker truncation** | 47% of pairs | bge-reranker-base reads 512 tokens, and about half of (question + chunk) pairs are longer, so the reranker sees a cut-off chunk. A longer-context reranker would help. |
| **Retrieval latency** | ~6.2 s per question | Measured on my 4 GB laptop GPU with the cross-encoder reranking the whole candidate pool plus the lane searches. |

11 questions still miss at least one required premise. Six of them get no targeted lane at
all, and two get regulator guidance where the binding article is required. One is a real gap in my corpus: the
only text saying US territories count as the US sits in a denied-boarding section that the
topic filter removes for a cancellation question.

---

## Generation: what works

10-question sample. `n` is the number of questions each metric applies to.

| metric | result | n | what it means here |
|---|---|---|---|
| **Citation validity** | **1.000** | 5 | Every `[Sn]` citation points to a source that was actually in the context. There were no invented citations. |
| **Citation coverage** | **1.000** | shown answers | Every answer shown to a user cited its factual sentences. The validator rejects an answer with an uncited legal, money or deadline claim before the user sees it. |
| **Clarification accuracy** | **1.000** | 1 | When the departure airport decides which law applies and the question doesn't give it, the bot asks once instead of guessing. |
| **Abstention accuracy** | **1.000** | 1 | Out-of-scope or trick questions get a decline, not an answer. |
| **Unsupported airline accuracy** | **1.000** | 1 | For an airline outside AA/DL/UA/WN the bot says so, and doesn't pretend to have that carrier's policy. |
| **Scope safety** | **1.000** | 1 | The bot never answers an out-of-scope question as if it were in scope. |
| **Over-abstention** | **0.000** | 8 | It never declined a question it should have answered. |
| **Truncated or oversized calls** | **0** | 13 calls | No answer was cut off by the token cap, and no request exceeded the model's window. |
| **Cost** | **$0.0007 per question** | 8 | Measured from token usage at Groq's list price. A full 44-question generation pass would cost about $0.03. |

The four safety gates are checked on one question each in this sample, so they show the
behaviour works. The same routing is checked on all 50 questions in the free
routing pass, which runs with no model.

## Generation: what's weak

| metric | result | n | what it means here, and why |
|---|---|---|---|
| **Factual correctness** | 0.50 (0.55 on answers delivered) | 8 (3) | The judge compares the answer's claims with my reference answer. Any question that produced no answer scores 0, which is what drags the first figure down. The references are my own drafts and aren't reviewed yet, so this is a consistency signal, not verified accuracy. |
| **Faithfulness** | 0.667 | 3 | The share of the answer's claims the judge could trace to the sources the model was actually given: the source blocks, the flight data and the prompt rules, never the reference answer. About a third of the claims weren't clearly supported by that evidence. |
| **Validation pass rate** | 0.600 | 5 | 2 of 5 answers still had an uncited factual sentence after one retry, so they were withheld rather than shown. That's the right behaviour for a legal-rights bot, but it means no answer. |

| **LLM error rate** | 0.375 | 8 | 3 of 8 calls failed with HTTP 429: Groq's free tier had hit its 200,000-token daily limit (199,429 used). This is a quota problem, not a model failure, but those 3 questions never got an answer. |
| **Generation time** | ~101 s mean | 8 | Measured in the eval harness, which spaces requests 45 s apart to stay under the free-tier token rate and includes the retry. It isn't what a user waits in the app. |

**The main takeaway: retrieval is no longer the bottleneck; citation discipline is.** After
the retrieval redesign, the evidence for these questions is in the context. The 20B model
often still writes a factual sentence without a citation, and the validator correctly
blocks it. I chose to keep the strict validator and lose those answers rather than show a
passenger an uncited claim about money they're owed.

---

## Next steps

- **Run all 50 questions through generation.** This needs either a paid tier or a resumable
  run that caches answers across days of free quota.
- **Get the reference answers reviewed**, so factual correctness measures accuracy rather
  than agreement with my drafts.
- **Try a stronger generator** for citation discipline, measured with the same frozen judge
  and the same retrieval.
