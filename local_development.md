# Local Development
This is how I run, index, monitor and evaluate the chatbot on my own machine. There are two ways to run it, and I test both:

- **On the host.** Fast, and uses the GPU. This is what I use day to day.
- **In Docker Compose.** The same image the Kubernetes cluster runs. Slower (CPU only), but
  it catches container problems before they reach AWS.

For the cloud deployment, see [AWS_deployment.md](AWS_deployment.md).

## Setup

All we need`.venv`, Docker Desktop and an `.env`. An NVIDIA GPU is
optional, but indexing takes about 100 s on a 4 GB GTX 1650 Ti against about 23 min on CPU.

`.env` is the only configuration file and my AIRLABS , GROQ, Google AI studio API keys live here.

python might not on PATH, so I call the venv directly
(`.venv/Scripts/python.exe`) or pass it to make: `make PYTHON=.venv/Scripts/python.exe <target>`.

## Run on the host

```powershell
scripts/run_local.ps1 -Index    # first run, or after a parser/schema change: rebuild the index
scripts/run_local.ps1           # just start the API
```

Then open http://127.0.0.1:8000.
I have given a  short video demo on README.md.

Then the script:

- Starts its own Postgres container, `fdr-app-pg`, on `127.0.0.1:55433`. It persists in the
  `fdr-app-pgdata` volume. `.env`'s `PG_DSN` points here.
- Clears any OS environment variable that `.env` also defines, so an old `$env:` value in
  the shell can't override the file.
- Runs the API with uvicorn on port 8000.

## Run with Docker Compose

```bash
docker compose build api
docker compose up -d postgres api
docker compose run --rm --no-deps -T api python scripts/index_corpus.py   # ~23 min on CPU
docker compose up -d prometheus grafana
```

| service | URL | notes |
|---|---|---|
| API | http://localhost:8000 | UI, `/ask`, `/health`, `/ready`, `/metrics` |
| Prometheus | http://localhost:9090 | `/targets` should show `flight-rights-api` UP |
| Grafana | http://localhost:3000 | dashboard provisioned from `deploy/grafana/dashboards/` |

Things to know:

- Port 8000 is usually taken by the host API. Use `API_PORT=8001 docker compose up -d api`.
- On a cold start the API downloads about 2.5 GB of model weights (bge-large-en-v1.5 and
  bge-reranker-base) into the `hfcache` volume. `/health` doesn't answer until that finishes,
  roughly 5 minutes.
- Everything binds to `127.0.0.1`. Grafana runs with anonymous admin access for convenience;
  that setting is for local use only.

The same steps are wrapped as `make up`, `make index-container` and `make down`.

## Health and readiness

```bash
curl http://127.0.0.1:8000/health    # process is up
curl http://127.0.0.1:8000/ready     # index exists and matches the running settings
```

`/ready` returning 503 is the readiness gate doing its job, not a crash. It stays 503 until
there's an index, and goes back to 503 if the index was built with a different parser,
embedding model or chunk size. The fix is to re-index.

## Indexing

```powershell
.venv/Scripts/python.exe scripts/index_corpus.py --dry-run   # parse the corpus only, no DB or embeddings
.venv/Scripts/python.exe scripts/index_corpus.py --reset     # full rebuild
```

The indexer writes to a staging table, validates it, and only then swaps it in, so a failed
run never leaves the app with a half-built index. Use `--reset` after a schema change.

## Monitoring

- The API exposes Prometheus metrics at `/metrics`.
- Alert rules are in `deploy/prometheus/rules/rag.yml`. Local Prometheus mounts that folder
  directly, so restart it after editing: `docker compose restart prometheus`.
- The cluster uses a generated copy of those rules. Regenerate it after every edit (a test
  fails if they drift):

  ```powershell
  .venv/Scripts/python.exe scripts/make_prometheusrule.py   # writes deploy/k8s/45-alert-rules.yaml
  ```

- On the dashboard, an empty panel means "no data", not zero. For example, cost only shows
  when the provider reports usage and `LLM_PRICE_*` is set.


## Tests and lint

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src scripts tests evals
```

## Evaluation

There are two stages, run in order, against a 50-case golden set:

1. **Retrieval** picks the chunk configuration. No LLM needed.
2. **Generation** answers the golden questions using that retrieval setup, and a separate (Gemini 3.5 Flash Lite)judge model scores them.

Neither stage calls AirLabs; flight data comes from fixtures.

### Golden set

```powershell
.venv/Scripts/python.exe evals/build_golden_set.py --check
```

The golden set is generated from the case specs in `evals/golden/`. Edit those, never the
JSONL.

### Stage 1: retrieval

This stage always runs on a throwaway Postgres + pgvector container, never on a database
holding real data:

```powershell
docker run -d --name fdr-test-pg --tmpfs /var/lib/postgresql/data:rw `
  -e POSTGRES_USER=fdr -e POSTGRES_PASSWORD=fdr -e POSTGRES_DB=fdr `
  -p 127.0.0.1:55432:5432 pgvector/pgvector:0.8.1-pg16

$env:EVAL_PG_DSN = "postgresql://fdr:fdr@127.0.0.1:55432/fdr"
.venv/Scripts/python.exe evals/retrieval_eval.py --sizes 256,512,1024
```

It takes about 15-20 minutes for three sizes on my GPU. The results go to
`evals/runs/retrieval/`, and `latest.json` is written only for a complete run, so an
interrupted sweep can't pass for a real result.

Useful flags:

- `--overlaps` sets the chunk overlaps to test.
- `--limit N` runs a quick subset.
- `--keep-schemas` leaves the indexes in the database for inspection.

### Stage 2: generation

The free pass needs no model and no judge. It checks routing and calibrates the confidence
gate:

```powershell
.venv/Scripts/python.exe evals/generation_eval.py --generator none
```

Runs against a real model use hosted APIs, so they cost money or free-tier quota. The script
refuses to send anything without `--confirm-paid-calls`. Without the flag it prints the plan
(cases, requests, pacing) and stops, and I read that plan first:

```powershell
# 2-case wiring check
.venv/Scripts/python.exe evals/generation_eval.py --generator groq-gpt-oss-20b --gate off --pilot --confirm-paid-calls

# 10-case sample, stratified by category so the clarify/abstain cases are always included
.venv/Scripts/python.exe evals/generation_eval.py --generator groq-gpt-oss-20b --gate off --sample 10 --confirm-paid-calls
```

Generator candidates are defined in `evals/generator_candidates.json`. The judge is set with
`JUDGE_*` in `.env`, and API keys are read by name and never written to results. The full
50-case run needs more tokens than Groq's free tier allows in a day, so I use the sample run
between full runs.

The results go to `evals/runs/generation/`.


## Troubleshooting

| symptom | cause / fix |
|---|---|
| `/ready` is 503 | No index, or one built with different settings. Run `scripts/run_local.ps1 -Index`, or `index_corpus.py --reset`. |
| Compose says port 8000 is in use | The host API is running. Use `API_PORT=8001 docker compose up -d api`. |
| First start takes minutes | Model weights are downloading. They're cached after that. |
| An alert rule change isn't showing | `docker compose restart prometheus`. |
| A setting seems ignored | An OS environment variable beats `.env`. Close the shell, or use `run_local.ps1`, which clears them. |

## Stopping

```bash
docker compose down          # keep the data
docker compose down -v       # also delete the database volume
docker rm -f fdr-test-pg     # remove the eval database
docker stop fdr-app-pg       # stop the host-mode database
```
