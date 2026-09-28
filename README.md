# Flight Delay RAG

> Active project: the core system is built, evaluated, and has been deployed on AWS. Feedback welcome.

<p align="center">
  <img src="docs/flight-delay-rag-demo.gif" width="100%" alt="Flight Delay RAG demo">
</p>

Your flight is delayed four hours. Are you owed $600, $520, a refund, a hotel, or nothing? Most passengers don’t know, because the answer can depend on where the flight departed, which airline operated it, and whether the airline’s own contract promises more than the law requires.

Flight Delay RAG is a chatbot came in my mind when my flight was delayed for 6 hours but I had to make it to the first day of class. That's been already over a year, I didn't know RAG then, AWS was just a buzzword to me. Now that I have started exploring world of platform Engineering, was like yeah..why not? Long story short, my project combines live flight status(AirLabsAPI) with text retrieved from government regulations (US DOT / 14 CFR, EU261, UK261) and
airline policies (American, Delta, United, Southwest). Every factual sentence in an answer
cites its source. I focused primarily on US domestic and Europe bound operations of these 4 carriers.

### What I had to get right

- **The route decides the law, so code decides the route, not even the LLM.** EU261 and UK261 apply to a US airline passenger only when the flight *departs* the EU or UK. London → New York can owe up to £520; New York → London owes no fixed compensation. Getting that wrong is the most expensive mistake the bot can make, so jurisdiction is resolved deterministically (EU261/UK261 by the departure airport, US DOT by whether the trip touches the US), and the governing law gets a guaranteed place in the model's context.

- **Government law and airline promises are kept apart.** Regulations say what a passenger is *entitled* to; contracts of carriage say what the airline *promised*. Retrieval reserves slots for both, and every source is labelled with LAW or AIRLINE in the prompt.

- **Ask, never guess.** If a question leaves out the detail about the flight that decides which law applies ("my flight was delayed 5 hours"), the bot asks one clarifying question about the flight no and destination (just the flight no good enough for AIRLABS API call). It never answers on a stated assumption.

- **No uncited claims.** A validator checks every sentence for a legal, money or deadline claim. Without a valid citation, it is sent back for one retry, and withheld if it still fails. No response is better than a confidently wrong one when money and lawsuits are involved.

## Demo

For the AWS architectural demo, refer to the [Deployment on AWS](#deployment-on-aws) section


## High-Level Architecture

<!-- Replace docs/architecture.png with the final architecture diagram. -->
<p align="center">
  <img src="docs/architecture.png" alt="High-level architecture diagram" width="900">
</p>

---

---

## Table of Contents

- [RAG Pipeline](#rag-pipeline)
  - [Evidence and indexing](#1-evidence-and-indexing)
  - [Question planning and route](#2-question-planning-and-route)
  - [Hybrid retrieval and reranking](#3-hybrid-retrieval-and-reranking)
  - [Generation and answer checks](#4-generation-and-answer-checks)
- [Evaluation](#evaluation)
- [Supported Scope](#supported-scope)
- [Tech Stack](#tech-stack)
- [Run Locally](#run-locally)
- [Repository Map](#repository-map)
- [Monitoring](#monitoring)
- [Deployment on AWS](#deployment-on-aws)
- [Testing](#testing)
- [Limitations](#limitations)

---

## RAG Pipeline

The difficult part of this project is retrieving the right legal premises for any specific route, then producing an answer that clearly shows where the information came from - law, airline policy, or live flight data. The index is built separately from passenger requests; the request path uses that verified index.

### 1. Evidence and indexing

The [`CORPUS` manifest](scripts/index_corpus.py) declares 17 source documents: EU261, UK261, U.S. DOT regulations and guidance, and policies from American, Delta, United, and Southwest. Each entry carries a document type, jurisdiction, publisher, and source URL. This metadata is used for filtering and for the citations shown to passengers.

[`ingest.py`](src/flight_delay/ingest.py) parses the source formats into sections and chunks within section boundaries. The selected configuration targets **512 tokens with 15% overlap**. [`index_corpus.py`](scripts/index_corpus.py) embeds the chunks with `BAAI/bge-large-en-v1.5` (1,024 dimensions), verifies that every required source produced chunks, and builds in `chunks_staging`. [`StagingIndex.activate`](src/flight_delay/store.py) validates the staged build and swaps it into the live `chunks` table in a database transaction. The recorded index identity includes the corpus digest, parser and chunk settings, and embedding model; [`/ready`](src/flight_delay/api.py) checks compatibility before an API pod serves traffic.

**Why this design:** a missing regulation or airline document should stop publication of the new index, rather than silently produce a plausible answer from incomplete evidence. A failed build leaves the previous live corpus available.

### 2. Question planning and route

[`plan_turn`](src/flight_delay/pipeline.py) classifies the request, carries forward conversation state when appropriate, extracts a carrier and flight number, and can look up a supported flight through AirLabs. The route is resolved in code. [`route_filters`](src/flight_delay/pipeline.py) separates **in-scope** regimes, whose text may be retrieved to explain applicability, from **governing** regimes, whose regulation gets a reserved opportunity to reach the final context.

When a missing airport would change the answer, the bot asks for it. It can also decline an unsupported carrier. Common follow-ups can reuse a prior validated answer without another retrieval and generation call. Live status is shown as flight data, never treated as the legal authority for compensation.

**Why this design:** for a US carrier, London → New York and New York → London can have different fixed-compensation rules. Letting the generator infer the applicable regime from whatever passages happened to rank highest would be a poor place to make that decision.

### 3. Hybrid retrieval and reranking

The live [`chunks` table](src/flight_delay/store.py) contains both pgvector embeddings and a generated PostgreSQL full-text column. [`HybridRetriever.retrieve`](src/flight_delay/retrieval.py) embeds the question, searches by cosine similarity, adds full-text matches missed by dense search, and adds targeted candidates for governing law and legal premises such as scope, remedy, or procedure. It reranks the resulting candidate pool with a local BGE cross-encoder, fuses ranking signals, and balances source types. The deployed main dense search requests **20 candidates**; the final context selects **up to 8 chunks**. Extra lexical and targeted-lane hits may enlarge the pool before reranking.

**Why both searches:** semantic search handles paraphrases; full-text search helps with exact legal references. **Why balancing:** a relevant airline page must not crowd out the binding regulation or a condition that determines whether it applies.

### 4. Generation and answer checks

[`build_context`](src/flight_delay/generation.py) admits guaranteed governing-law chunks before spending the token budget on other sources. It labels each source as a regulation, regulator guidance, or airline policy, assigns `[S1]`-style markers, and keeps a map from each marker to the actual section and URL. The default generator is hosted Groq `openai/gpt-oss-20b`; the API pods run embedding and reranking locally on CPU.

After generation, [`validate_answer`](src/flight_delay/generation.py) rejects invented source markers and factual-looking sentences without citations. [`RagPipeline.run`](src/flight_delay/pipeline.py) can send a targeted retry note and withholds an answer that still fails its checks, unless removing whole uncited bullet points leaves an answer that passes them with at least one citation. The validator checks **citation structure and coverage**, not whether a cited passage truly supports a claim; factual support is evaluated separately. A confidence gate exists in code, but the [deployed ConfigMap](deploy/k8s/18-config.yaml) has it **disabled**.

## Evaluation

The [golden set](evals/golden_set.jsonl) contains 50 cases, including answerable routes, ambiguous requests, unsupported carriers, and traps. Retrieval is scored on the answerable cases; generation has a separate evaluation stage.

| Stage | What was measured | Current status |
|---|---|---|
| [Retrieval](evals/retrieval_eval.py) | Required-premise completeness, primary-authority coverage, evidence recall, ranking | Latest selected run: **32/43** answerable cases have every required premise; evidence recall **0.571**, MRR **0.647**. Eleven cases still miss a premise. |
| [Generation](evals/generation_eval.py) | Faithfulness, factual correctness, citations, clarification and abstention | A 10-case sample exists. The authoritative full 50-case run **has not completed**, so its results are provisional. |

The retrieval numbers measure **evidence available to the generator**, not the accuracy of 32 final answers. See [`RESULTS.md`](RESULTS.md) and the [latest retrieval run](evals/runs/retrieval/latest.json) for definitions and run records. The full generation evaluation remains an open task.

## Supported Scope

| | Coverage |
|---|---|
| Airlines | American (AA), Delta (DL), United (UA), Southwest (WN) |
| Routes | US domestic and these carriers' EU/UK operations |
| Legal sources | U.S. DOT / 14 CFR, EU261, UK261, plus regulator guidance |

A flight outside the covered route or carrier set is not evidence that a particular foreign law applies. The assistant asks for missing route details or refers the passenger to the relevant authority where its sources do not establish an answer.

## Tech Stack

| Purpose | Implementation |
|---|---|
| API and browser UI | FastAPI |
| Retrieval store | PostgreSQL with pgvector and built-in full-text search |
| Embeddings and reranking | `bge-large-en-v1.5`; `bge-reranker-base` |
| Answer generation | Groq `openai/gpt-oss-20b` through an OpenAI-compatible API |
| Live flight status | AirLabs API |
| Evaluation | Custom golden-set retrieval metrics; Ragas-assisted generation evaluation |
| Local and cloud deployment | Docker Compose; AWS EKS with Terraform, eksctl, and Helm |

## Run Locally

### Prerequisites

- Python 3.11+ and [uv](https://github.com/astral-sh/uv)
- Docker for PostgreSQL and the container stack
- An AirLabs API key and an OpenAI-compatible LLM endpoint
- Optional CUDA GPU for faster local embedding

### Configuration

Local settings live in an uncommitted `.env` at the repository root: database DSN, model endpoint and key, AirLabs key, and retrieval/generation settings. See `.env.example` for the required names.

### From the repository root

```bash
make install        # core dependencies
make install-ml     # embedder + reranker
make db             # start Postgres
make index          # build and activate the corpus index
make run            # API and browser UI at http://localhost:8000
```

### Docker Compose

```bash
make up                 # postgres + api + prometheus + grafana
make index-container    # build the index using the application image
```

## Repository Map

| Path | Responsibility |
|---|---|
| [`scripts/index_corpus.py`](scripts/index_corpus.py), [`src/flight_delay/ingest.py`](src/flight_delay/ingest.py) | Source manifest, parsing, chunking, index build |
| [`src/flight_delay/store.py`](src/flight_delay/store.py) | pgvector/full-text SQL, staged activation, conversation state |
| [`src/flight_delay/pipeline.py`](src/flight_delay/pipeline.py), [`src/flight_delay/retrieval.py`](src/flight_delay/retrieval.py) | Request planning, route filters, candidate search, reranking |
| [`src/flight_delay/generation.py`](src/flight_delay/generation.py), [`system_prompt.md`](system_prompt.md) | Context assembly, generator contract, citation checks |
| [`evals/`](evals/) | Golden cases, retrieval and generation evaluations |
| [`deploy/`](deploy/) | Kubernetes, Helm, Terraform, Prometheus, Grafana |

## Monitoring

The API exports Prometheus metrics from [`src/flight_delay/metrics.py`](src/flight_delay/metrics.py). The default Grafana dashboard is [`deploy/grafana/dashboards/rag.json`](deploy/grafana/dashboards/rag.json). Local Docker Compose provisions monitoring; the EKS monitoring stack is installed separately with Helm. Alert rules track RAG failures as well as infrastructure signals.

## Deployment on AWS

### AWS Deployment Demo

https://github.com/user-attachments/assets/21752488-15c2-477c-a57d-8aa5eb7cf89e

### Deployment Path

<p align="center">
  <img src="docs/deployment_map.png"
       width="100%"
       alt="AWS EKS deployment and connection map">
</p>

### Deployment & Connection Map Walkthrough

[![Watch the deployment walkthrough](https://img.youtube.com/vi/JBMAd6md55k/maxresdefault.jpg)](https://www.youtube.com/watch?v=JBMAd6md55k)

▶ [Watch the full deployment and connection map explanation on YouTube](https://www.youtube.com/watch?v=JBMAd6md55k)

In this walkthrough, I explain how the deployment fits together behind the scenes, from the Docker containers to the Kubernetes resources running on EKS. I also go through how the API, PostgreSQL/pgvector, Services, NetworkPolicies, load balancer, and monitoring components connect, and what role each one plays in the overall architecture.
Full step-by-step procedure: [AWS_deployment.md](AWS_deployment.md).

## Testing

```bash
make test    # pytest
make lint    # ruff
```

## Limitations

- Covers four US carriers and three jurisdictions only.
- Routes departing outside the US, EU and UK are referred to the relevant authority, not answered.
- Generation results are provisional until the full 50-case Stage 2 run completes.
- This assistant gives information, not legal advice, and cannot book or change flights.
