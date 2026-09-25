# Local Build Runbook

This file covers the full local workflow. There are two independent ways to run
it - **on the host** and **in containers** - and they are verified separately,
because each has failure modes the other does not.

> `make` targets call `python`, which is not on PATH on this Windows machine.
> Either pass the interpreter (`make PYTHON=.venv/Scripts/python.exe <target>`)
> or run the underlying command directly, as shown throughout this file.

## A. On the host (the fast path: uses the GPU)

```powershell
scripts/run_local.ps1 -Index    # first time, or after a parser change (~100 s on a GTX 1650 Ti)
scripts/run_local.ps1           # start the API; open http://127.0.0.1:8000
```

Every setting comes from `.env`, the project's only configuration file (its
`PG_DSN` points at the `fdr-app-pg` container on :55433, which the script starts).
The script sets no settings of its own and clears any leftover OS environment
variable that `.env` defines, so an old `$env:` value cannot override it.

## B. In containers (what the cluster actually runs)

```bash
docker compose build api
docker compose up -d postgres api
docker compose run --rm --no-deps api python scripts/index_corpus.py   # ~23 min: CPU
docker compose up -d prometheus grafana
```

- **Port 8000 is often already taken** by the host API above. The compose port is
  configurable: `API_PORT=8001 docker compose up -d api`.
- The API container downloads ~2.5 GB of bge weights on first start into the
  `hfcache` volume; `/health` does not answer until that finishes (~5 min cold).
- `/ready` returns **503 `index is empty`** until the index exists, and 503 again
  if the index disagrees with the running settings. That is the readiness gate,
  not a bug: a pod with no index must not answer.
- Indexing in the container is CPU-only: 21m50s measured, ~13x slower than the
  ~100 s the host GPU takes. It is worth doing at least once, because it is exactly what the
  Kubernetes index Job runs.

| service | URL | note |
|---|---|---|
| API | http://localhost:8000 (or `$API_PORT`) | UI + `/ask`, `/health`, `/ready`, `/metrics` |
| Prometheus | http://localhost:9090 | `/targets` must show `flight-rights-api` UP |
| Grafana | http://localhost:3000 | dashboard provisioned from `deploy/grafana/dashboards/` |

Both images are pinned (`prom/prometheus:v3.14.0`, `grafana/grafana:13.2.2`);
Grafana's anonymous-admin setting is a local-demo convenience and must never be
copied to a public deployment.

### Reading the dashboard

An empty panel means **unavailable**, not zero: cost is only emitted when the
provider reports usage *and* `LLM_PRICE_*` are set, and gate-refusal series stay
empty while `CONFIDENCE_GATE_ENABLED=false`. Retrieval quality
(`evidence_recall`, `section_recall@3`, MRR) is deliberately **not** on any
dashboard - the application does not compute it; `evals/retrieval_eval.py` does,
offline, against the golden set.

Alert thresholds live in `deploy/prometheus/rules/rag.yml`. After editing them,
regenerate the cluster copy so the two cannot drift:

```powershell
.venv/Scripts/python.exe scripts/make_prometheusrule.py
```

## Evaluation

Two stages, in order. Stage 1 chooses the chunk configuration; stage 2 generates
answers at that configuration and judges them. Neither calls AirLabs.

```bash
make golden-check     # validate the 50-case golden set
make eval-retrieval   # stage 1 -> evals/runs/retrieval/latest.json (needs EVAL_PG_DSN)
make eval-generation  # stage 2 -> evals/runs/generation/latest.json
make eval             # both
```

### Stage 1 on PostgreSQL (Windows PowerShell)

Stage 1 is authoritative only on PostgreSQL + pgvector, the production store. Use a
DISPOSABLE database, never the compose database that holds conversations. The
evaluation only creates and drops schemas named `eval_retrieval_*`.

```powershell
docker run -d --name fdr-test-pg --tmpfs /var/lib/postgresql/data:rw `
  -e POSTGRES_USER=fdr -e POSTGRES_PASSWORD=fdr -e POSTGRES_DB=fdr `
  -p 127.0.0.1:55432:5432 pgvector/pgvector:0.8.1-pg16
$env:EVAL_PG_DSN = "postgresql://fdr:fdr@127.0.0.1:55432/fdr"
.venv/Scripts/python.exe evals/build_golden_set.py --check
.venv/Scripts/python.exe evals/retrieval_eval.py --sizes 256,512,1024
docker rm -f fdr-test-pg          # when finished (tmpfs: nothing persists)
```

First run downloads BAAI/bge-large-en-v1.5 and BAAI/bge-reranker-base (~2.5 GB).
On a 4 GB GTX 1650 Ti the three sizes take roughly 15-20 minutes. `--keep-schemas`
leaves each index in the database for inspection; `--limit N` makes a smoke run.

Stage 1 takes `--sizes`, `--overlaps`, `--limit`, `--backend`, `--pg-dsn` and
`--keep-schemas`; stage 2 takes `--retrieval-result PATH` if you do not want
`latest.json`. An offline smoke run (numbers explicitly labelled as not
measurements; it writes only `latest-smoke.json`):

```powershell
$env:ALLOW_TEST_DOUBLES = "true"
.venv/Scripts/python.exe evals/retrieval_eval.py --backend memory --embedder hash --reranker none --sizes 256,512 --limit 10
.venv/Scripts/python.exe evals/generation_eval.py --backend memory --embedder hash --reranker none --generator echo --judge fake --limit 10 --allow-smoke-selection --retrieval-result evals/runs/retrieval/latest-smoke.json
```

### Stage 2 on PostgreSQL: gate calibration and the generator bake-off (PowerShell)

Stage 2 reads `evals/runs/retrieval/latest.json`, indexes the selected configuration
into the schema `eval_generation_<size>_<overlap>` of the same DISPOSABLE database
(dropped afterwards; `--keep-schema` keeps it) and holds retrieval fixed: every
generator candidate answers from identical chunks and the same request budget.

1. **Free: routing and confidence-gate calibration.** No model, no judge, no API
   call. Writes `evals/runs/generation/calibration-<stamp>.json` (never `latest.json`).

   ```powershell
   $env:EVAL_PG_DSN = "postgresql://fdr:fdr@127.0.0.1:55432/fdr"
   .venv/Scripts/python.exe evals/generation_eval.py --generator none
   ```

2. **Keys and judge** in `.env` (read by the script; never printed or written to a
   result file). The judge is ONE model for the whole run and must not be a candidate:

   ```
   JUDGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   JUDGE_MODEL=gemini-3.5-flash-lite  # frozen judge (JUDGE_EXPECTED_MODEL); another model = override
   JUDGE_API_KEY=<Google AI Studio key>   # judge calls only
   GROQ_API_KEY=<Groq key>            # generator calls only (candidates' api_key_env); never
                                      # replaced by JUDGE_API_KEY.
   ```

   The judge moved off Groq in Session 19 for an arithmetic reason, not a
   preference: a 50-case judge pass needs ~733k tokens against Groq's 200k/day
   free cap, and gpt-oss-120b's 8,000 TPM is smaller than one judge request
   reserving its 8,192-token cap (HTTP 413, never retried). AI Studio caps
   *requests* per day (500 for Flash-Lite; every full Flash model there is RPD
   20, i.e. unusable) with no token-per-day cap.

   The judge's request is frozen in `generation_eval.py` (`JUDGE_REQUEST`):
   `temperature` 0.0, cap 8192 sent as `max_tokens`, instructor JSON mode, and
   **no `reasoning_effort`** (a Groq/OpenAI field Gemini rejects). `JUDGE_PRICES`
   is `None`, so judge cost reports as *unavailable* rather than as zero. Calls
   are paced at 4.2 s (<= 14 req/min), and `preflight_judge()` makes one
   structured call **before** indexing or generation, so an unreachable judge
   costs no generator tokens.

3. **Candidates** are named in `evals/generator_candidates.json` (provider, endpoint,
   model, served `context_window`, `api_key_env`, `extra_body`, `max_completion_tokens`,
   `provider_max_completion_tokens`, `min_call_interval_s`, `retry_max_wait_s` and prices).
   The candidate is `groq-gpt-oss-20b` (`reasoning_effort` low) with
   `max_completion_tokens` 900 (sent as `max_completion_tokens`; reasoning
   included), paced **45 s** between request starts - derived, not guessed: free-tier
   TPM 8000 refills at ~133 tok/s and one call costs ~5,700. `groq-qwen3.8-27b` was
   retired by the user in Session 19 and remains in the file only as a record, so
   Stage 2 is now a single-candidate **acceptance test** against the hard gates,
   not a bake-off: there is nothing left to rank.
   Retry-After is honoured up to 120 s, and a `request_too_large` refusal is never
   retried. The common **input-assembly
   budget** (`LLM_CONTEXT_WINDOW` 9728, frozen from the selection) with its
   `CONTEXT_ANSWER_RESERVE_TOKENS` (700, formerly `LLM_MAX_TOKENS`) sizes the sources, so every
   candidate sees the same ones; it is not a model window. Each call records
   prompt/completion/reasoning tokens, finish_reason, truncation and any HTTP error (status,
   provider code, complete sanitized message, Retry-After), and checks
   `prompt_tokens + cap <= context_window` (violations are printed).

4. **Pilot, subset, full run.** Any remote generator or judge is refused unless
   `--confirm-paid-calls` is given; the plan (endpoints, requests, cases) is printed first.
   `--pilot` runs the fixed cases `PILOT_CASE_IDS` (uk-us-01, a complex answer; trap-04, an
   out-of-scope edge case) to validate wiring only: it never selects a generator, and it
   extrapolates the full run's generator + judge cost from its measured usage. The routing
   pass always covers all 50 cases. Subset runs write `latest-smoke.json`.

   `--sample N [--sample-seed S]` draws N cases **stratified by category** - a uniform
   draw of 10 omits the clarify or abstain cases about half the time, and those are the
   hard gates (an empty gate passes vacuously and reads as success). The seed is printed
   and recorded so a draw repeats. A subset run may select a generator (user's decision,
   Session 19) but stays non-authoritative: its recommendation carries a `basis` string
   and `full_run: false`.

   ```powershell
   # wiring only, 2 cases
   .venv/Scripts/python.exe evals/generation_eval.py --generator groq-gpt-oss-20b --gate off --pilot --confirm-paid-calls
   # 10-case stratified subset: provisional selection
   .venv/Scripts/python.exe evals/generation_eval.py --generator groq-gpt-oss-20b --gate off --sample 10 --confirm-paid-calls
   # the authoritative 50-case run - needs ~2.5 days of Groq free-tier TPD
   .venv/Scripts/python.exe evals/generation_eval.py --generator groq-gpt-oss-20b --gate off --confirm-paid-calls
   ```

   Leave out `--confirm-paid-calls` to print the plan and stop before anything is sent.
   **Ask the user before every paid run**, every time.

   Budget arithmetic that decides which of these you can afford today: one call is
   ~5,700 tokens (the system prompt alone is ~2,3xx of it, 42% and fixed), a 50-case
   run needs ~285k without retries and ~493k at the observed retry rate, against a
   **200,000 token/day** free cap. A single-day full run is arithmetically impossible
   on the free tier at any `final_k`.

What it reports per candidate: Ragas faithfulness **to authorized evidence** (source
blocks as sent, the route/airline/flight notes and FLIGHT DATA sent with them, and
system_prompt.md Rules 1, 2, 4, 5; never the reference) over shown generated answers;
Ragas factual correctness over **every** answer-required case (no answer delivered =
0.0, listed) plus the judged-only mean; outcome counts; abstention, clarification and
unsupported-airline accuracy; validation and first-try pass rates; LLM error rate;
citation validity and coverage; scope safety; generator and judge tokens, cost and latency.
A recommendation follows the printed `SELECTION_RULE` (hard gates, then Faithfulness within
0.05 of best, FactualCorrectness, cost/case, tokens/case, latency; no eligible candidate = no
winner), only on a full run, and is never applied. With a winner it prints the explicit
production settings, including `LLM_MAX_COMPLETION_TOKENS` (the tested cap), `LLM_EXTRA_BODY`,
`CONTEXT_ANSWER_RESERVE_TOKENS` and `LLM_CONTEXT_WINDOW`; the app refuses a reasoning
`LLM_EXTRA_BODY` without an explicit `LLM_MAX_COMPLETION_TOKENS`. Every expected answer is
still an unvetted draft.

Adopting the selected chunk size in the customer-facing index is a deliberate
separate step: set `CHUNK_TARGET_TOKENS` / `CHUNK_OVERLAP_PCT` and re-index.
