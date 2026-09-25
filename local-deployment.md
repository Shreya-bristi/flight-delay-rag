# Local Development Runbook

This runbook covers local development, indexing, monitoring, and evaluation for the flight-delay RAG application.

## Prerequisites

- Python virtual environment at `.venv`
- Docker Desktop running
- `.env` configured
- Optional NVIDIA GPU for faster host-side indexing

> **Note:** `make` targets call `python`. On Windows, either pass `PYTHON=.venv/Scripts/python.exe` or run the underlying command directly.

## 1. Run locally on the host

```powershell
scripts/run_local.ps1 -Index    # rebuild the corpus index, then start the API
scripts/run_local.ps1           # start the API without re-indexing
```

Open:

- API/UI: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`
- Readiness: `http://127.0.0.1:8000/ready`
- Metrics: `http://127.0.0.1:8000/metrics`

The script uses `.env` as the local configuration source and starts the dedicated Postgres container `fdr-app-pg` on `127.0.0.1:55433`.

## 2. Run with Docker Compose

```bash
docker compose build api
docker compose up -d postgres api
docker compose run --rm --no-deps api python scripts/index_corpus.py
docker compose up -d prometheus grafana
```

If port 8000 is already in use:

```bash
API_PORT=8001 docker compose up -d api
```

| Service | Address | Purpose |
|---|---|---|
| API | `http://localhost:8000` | UI and application endpoints |
| Prometheus | `http://localhost:9090` | Metrics and alert evaluation |
| Grafana | `http://localhost:3000` | Local monitoring dashboard |

> **Note:** The API downloads embedding and reranking model weights on first startup, so a cold start can take several minutes.

`/ready` returns `503` until the corpus index exists and matches the running configuration.

## 3. Rebuild the corpus index

```powershell
.venv/Scripts/python.exe scripts/index_corpus.py --reset
```

Dry run:

```powershell
.venv/Scripts/python.exe scripts/index_corpus.py --dry-run
```

The indexer builds in staging, validates the result, and activates it only after validation succeeds.

## 4. Verify the application

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

- `/health` confirms the process is alive.
- `/ready` confirms the application has a compatible index and can serve requests.

## 5. Local monitoring

Prometheus rules:

```text
deploy/prometheus/rules/rag.yml
```

After changing them:

```powershell
.venv/Scripts/python.exe scripts/make_prometheusrule.py
```

Generated Kubernetes rule:

```text
deploy/k8s/45-alert-rules.yaml
```

Grafana dashboards:

```text
deploy/grafana/dashboards/
```

Retrieval-quality metrics such as recall, MRR, and nDCG are evaluated offline rather than exported by the application.

## 6. Evaluation

Evaluation has two stages:

1. Retrieval evaluation selects the retrieval/chunking configuration.
2. Generation evaluation measures answer quality using that retrieval setup.

### 6.1 Validate the golden set

```powershell
.venv/Scripts/python.exe evals/build_golden_set.py --check
```

### 6.2 Retrieval evaluation

Use a disposable PostgreSQL + pgvector database:

```powershell
docker run -d --name fdr-test-pg --tmpfs /var/lib/postgresql/data:rw `
  -e POSTGRES_USER=fdr `
  -e POSTGRES_PASSWORD=fdr `
  -e POSTGRES_DB=fdr `
  -p 127.0.0.1:55432:5432 `
  pgvector/pgvector:0.8.1-pg16

$env:EVAL_PG_DSN = "postgresql://fdr:fdr@127.0.0.1:55432/fdr"

.venv/Scripts/python.exe evals/retrieval_eval.py --sizes 256,512,1024
```

Cleanup:

```powershell
docker rm -f fdr-test-pg
```

Results:

```text
evals/runs/retrieval/
```

### 6.3 Generation evaluation

Calibration:

```powershell
$env:EVAL_PG_DSN = "postgresql://fdr:fdr@127.0.0.1:55432/fdr"
.venv/Scripts/python.exe evals/generation_eval.py --generator none
```

Typical hosted-model sample:

```powershell
.venv/Scripts/python.exe evals/generation_eval.py `
  --generator groq-gpt-oss-20b `
  --gate off `
  --sample 10 `
  --confirm-paid-calls
```

Results:

```text
evals/runs/generation/
```

> **Important:** Remote generator or judge calls require `--confirm-paid-calls`. Review the printed plan before running them.

## 7. Smoke evaluation

```powershell
$env:ALLOW_TEST_DOUBLES = "true"

.venv/Scripts/python.exe evals/retrieval_eval.py `
  --backend memory `
  --embedder hash `
  --reranker none `
  --sizes 256,512 `
  --limit 10

.venv/Scripts/python.exe evals/generation_eval.py `
  --backend memory `
  --embedder hash `
  --reranker none `
  --generator echo `
  --judge fake `
  --limit 10 `
  --allow-smoke-selection `
  --retrieval-result evals/runs/retrieval/latest-smoke.json
```

Smoke results are non-authoritative.

## 8. Common issues

### `/ready` returns 503

Rebuild the index:

```powershell
.venv/Scripts/python.exe scripts/index_corpus.py
```

If it still fails, check index compatibility with the current parser, embedding model, and chunk settings.

### Port 8000 is already in use

```bash
API_PORT=8001 docker compose up -d api
```

### First startup is slow

The embedding and reranking model weights are downloaded on first use and cached afterwards.

### Prometheus alerts are missing

Regenerate the Kubernetes rule manifest:

```powershell
.venv/Scripts/python.exe scripts/make_prometheusrule.py
```

## 9. Stop local services

```bash
docker compose down
```

Remove the disposable evaluation database if needed:

```bash
docker rm -f fdr-test-pg
```
