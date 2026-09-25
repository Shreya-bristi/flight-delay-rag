#!/usr/bin/env python3
"""
Stage 2 of 2: generate answers at the selected chunk configuration, judge them,
and compare generator candidates on identical retrieval (the bake-off).

USAGE (PowerShell; the database must be a DISPOSABLE pgvector instance, see RUNBOOK.md)
    $env:EVAL_PG_DSN = "postgresql://fdr:fdr@127.0.0.1:55432/fdr"
    # free: retrieval, routing and confidence-gate calibration only, no model and no judge
    .venv/Scripts/python.exe evals/generation_eval.py --generator none
    # the bake-off: candidates by name from evals/generator_candidates.json
    .venv/Scripts/python.exe evals/generation_eval.py --generator groq-gpt-oss-20b --generator groq-qwen3.8-27b --gate off --pilot --confirm-paid-calls
    # offline smoke (no database, no model, no judge)
    .venv/Scripts/python.exe evals/generation_eval.py --backend memory --embedder hash --reranker none --generator echo --judge fake --retrieval-result evals/runs/retrieval/latest-smoke.json --allow-smoke-selection

This is the ONLY generation-quality evaluator. It reads the chunk configuration
chosen by evals/retrieval_eval.py (evals/runs/retrieval/latest.json) and refuses a
missing, incomplete, smoke or stale selection. It builds ONE index at that
configuration, on PostgreSQL (the production store; schema
eval_generation_<size>_<overlap>, dropped afterwards) or in memory for a smoke run,
and holds retrieval fixed: every candidate answers from the same retrieved chunks,
the same built context and the same evidence budget.

PASSES
------
0. Routing pass, no model: every golden case through the production pipeline with
   a generator that stops once the context is built. It records each turn's
   structured outcome, the confidence gate's decision and signals, and warms the
   retrieval cache. Its gate data is the calibration report (below). With
   `--generator none` the run stops here and costs nothing.
1. One pass per generator candidate, same cases, same retrieval, same gate.
2. The judge (one model, frozen for the whole run) scores every candidate.

WHAT IS MEASURED (per candidate)
--------------------------------
Two Ragas LLM-judged metrics, and no others:

  faithfulness (to authorized evidence)
      The shown answer against the evidence generation was AUTHORIZED to use:
      the numbered source blocks exactly as sent (headers included, taken from the
      turn's BuiltContext, never rebuilt), the notes and FLIGHT DATA block sent with
      them (route assumption, airline notice, flight-data note), and the substantive
      rules of the unchanged system_prompt.md (Rules 1, 2, 4, 5: hierarchy, route,
      delay cause, entitlement types), each labelled as what it is. The reference
      answer is never part of it. Applies to every generated answer that was shown
      (outcome answered or abstained); the denominator is printed.
  factual_correctness
      The model's answer (without the code-written openers) against the golden
      reference, over EVERY answer-required case. A case where no answer was
      delivered (gated, validation_failed, llm_error, still clarifying, declined)
      scores 0.0 and is listed, so a retrieval or generation failure can never
      improve the mean by disappearing. The judged-only mean is reported beside it.

Deterministic metrics, from the structured Answer.outcome (never from prose):
  outcome counts; abstention_accuracy (abstain cases declined), clarification_accuracy
  (clarify cases asked, once), unsupported_airline_accuracy (never looked up, no
  supported carrier's documents, carrier flagged), over_abstention_rate (answer cases
  gated or declined by the model), validation_pass_rate and first_try_pass_rate,
  llm_error_rate, citation_validity (the model's final output names only supplied
  sources), citation_coverage, tokens, cost when prices are configured, latency.
  Every metric carries its denominator and failing case ids.

GATE CALIBRATION (generator independent, from pass 0)
  Per case: the gate's decision and signals (top rerank score, score margin,
  lexical grounding). Answer-required cases the gate refuses are false abstentions;
  abstain cases it lets through are missed abstentions; clarify cases are neither.
  A threshold grid reports both counts per setting. Nothing is changed: thresholds
  live in config.py and are the user's decision.

BAKE-OFF RECOMMENDATION (printed, never applied; full runs only - see SELECTION_RULE)
  hard gates: generation errors <= 5%, judge errors <= 10% after retries, zero
  fabricated/unknown citations, zero incorrect unsupported-airline/out-of-scope
  answers, every clarification and abstention edge case passes
  -> Faithfulness within 0.05 of the best eligible -> highest FactualCorrectness
  -> lowest measured cost/case -> tokens/case -> latency. No eligible candidate =
  no winner. The winner's explicit production settings (its tested completion cap
  included) are printed with it.

PILOT (--pilot = PILOT_CASE_IDS; --cases for other fixed ids)
  Validates wiring on fixed representative cases and never selects. The routing
  pass still covers every case, so the full run's generation and judge cost is
  extrapolated from the pilot's MEASURED per-case usage.

PAID CALLS
  Any generator or judge endpoint that is not on this machine is a paid or
  rate-limited API. The run prints its plan (endpoints, cases, calls) and refuses
  to start without --confirm-paid-calls. Keys are read from the environment or .env
  and are never printed or written to a result file.

REFERENCES ARE DRAFTS
  The golden expected answers have not been vetted by a domain expert, so
  factual_correctness is a consistency signal, not validated accuracy. Every run
  says so and records it.

AIRLABS
  No evaluation path calls AirLabs: every case runs through
  golden.replay.run_conversation with its own FixtureTool.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "evals"):
    sys.path.insert(0, str(p))

# Stage 1 owns the golden-set loader, the digests and the index builders;
# importing them keeps the two stages provably on the same inputs.
from retrieval_eval import (  # noqa: E402  (sys.path set above)
    LATEST as RETRIEVAL_LATEST,
)
from retrieval_eval import (  # noqa: E402
    atomic_write_text,
    build_index,
    build_pg_index,
    corpus_digest,
    display_path,
    drop_schema,
    golden_digest,
    golden_version,
    load_golden_rows,
    pg_versions,
    prompt_digest,
)

OUT_DIR = ROOT / "evals" / "runs" / "generation"
LATEST = OUT_DIR / "latest.json"
LATEST_SMOKE = OUT_DIR / "latest-smoke.json"
CANDIDATES_FILE = ROOT / "evals" / "generator_candidates.json"
SCHEMA_PREFIX = "eval_generation"

# Judge configuration, read from the environment (or .env) so a hosted key never
# has to be typed on a command line or stored in a result file.
JUDGE_ENV_BASE_URL = "JUDGE_BASE_URL"
JUDGE_ENV_MODEL = "JUDGE_MODEL"
JUDGE_ENV_API_KEY = "JUDGE_API_KEY"

# The judge, frozen for every candidate (user's decision, Session 14): the model is
# JUDGE_MODEL (expected below; a different one is recorded as an override and has
# no price), and this request goes with every judge call.
#
# Session 19 (2026-09-17), user's decision: the judge moved from Groq
# openai/gpt-oss-120b to Google AI Studio gemini-3.5-flash-lite, because Groq's free
# tier cannot judge this golden set. Measured from the Session 17 run: a full 50-case
# judge pass needs ~733,000 tokens against a 200,000 tokens-per-day cap (3.7 days),
# and gpt-oss-120b's 8,000 tokens-per-MINUTE cap is smaller than a single judge
# request reserving its own 8,192-token completion cap, which returned HTTP 413
# request_too_large and is never retried. On AI Studio's free tier this model is
# capped on REQUESTS per day (500) with no token-per-day cap at all, and TPM is
# 250,000 counted on input only, so neither limit binds here.
#
# reasoning_effort is deliberately absent: it is a Groq/OpenAI field, and Gemini's
# OpenAI-compatibility layer does not accept it. Thinking level is left at the
# model's default. temperature 0 so the judge does not vary between candidates. The
# cap goes out under the host's own field name (cap_parameter): max_completion_tokens
# for Groq, max_tokens elsewhere.
JUDGE_EXPECTED_MODEL = "gemini-3.5-flash-lite"
JUDGE_REQUEST = {"temperature": 0.0, "max_completion_tokens": 8192}
JUDGE_STRUCTURED_OUTPUT = ("instructor Mode.JSON via Ragas llm_factory(provider='openai'): "
                           "response_format {'type': 'json_object'}, the Pydantic schema written "
                           "into the prompt, instructor re-asks on invalid JSON")
# No published price for this model is recorded in this repository, so judge cost is
# reported as unavailable rather than as zero (the same rule generator_candidates.json
# uses for a candidate with null prices). Applied only to JUDGE_EXPECTED_MODEL.
JUDGE_PRICES = None
# AI Studio free tier for gemini-3.5-flash-lite: RPM 15, TPM 250K (input), RPD 500.
# The judge is scored one case at a time, but a case makes several calls back to back,
# so the client paces itself to stay under RPM. 60/15 = 4.0 s; 4.2 s leaves a margin
# for clock skew. RPD resets at midnight Pacific.
JUDGE_MIN_CALL_INTERVAL_S = 4.2
JUDGE_RPD_LIMIT = 500

# Fixed pilot cases (user's rule, Session 14): wiring only, never used to choose.
#   uk-us-01  complex answer: UK->US, UK261 + US DOT + United's contract, multi-chunk
#   trap-04   edge: out-of-scope (Canada APPR), one clarification round, then it must decline
PILOT_CASE_IDS = ("uk-us-01", "trap-04")

EVIDENCE_SCOPE = (
    "faithfulness to AUTHORIZED evidence: the numbered source blocks exactly as sent to "
    "the model, the route/airline/flight notes and FLIGHT DATA block sent with them, and "
    "system_prompt.md Rules 1, 2, 4 and 5; never the reference answer")
# Rules of system_prompt.md whose content is substantive guidance the model may
# state (legal hierarchy, which law a route triggers, delay-cause categories,
# entitlement types and thresholds). Rules 3, 6 and 7 are citation, format and
# abstention instructions: nothing an answer could claim as a fact.
PROMPT_GUIDANCE_RULES = (1, 2, 4, 5)

# Outcomes (models.Outcome) and what they mean for the metrics.
SHOWN_GENERATED = ("answered", "abstained")          # model text the passenger saw
DECLINED = ("abstained", "gated", "declined_unsupported")

SELECTION_RULE = (
    "hard gates: generation errors <= 5% and judge errors <= 10% after retries; zero "
    "fabricated/unknown citations; zero incorrect answers on unsupported-airline/out-of-scope "
    "cases; every preserved clarification and abstention edge case passes. Among eligible: "
    "Faithfulness within 0.05 of best -> highest FactualCorrectness (all answer-required cases, "
    "no answer = 0) -> lowest measured cost/case -> tokens/case -> latency. No candidate "
    "passing the hard gates = no winner. Applied only to a full run (user's rule, Session 14)")
MAX_LLM_ERROR_RATE = 0.05
MAX_JUDGE_ERROR_SHARE = 0.10
FAITHFULNESS_TOLERANCE = 0.05


# --------------------------------------------------------------------------
# The selected retrieval configuration
# --------------------------------------------------------------------------


def load_retrieval_selection(path: Path, allow_smoke: bool = False) -> dict:
    """
    The handoff from stage 1. Refuses anything that is not a finished selection.

    A half-written or stale file is the one failure that would silently
    invalidate every number below it: the answers would be generated at a chunk
    size nobody chose, against gold labels, sources or a prompt that no longer
    match. A smoke selection (stand-in models or --limit) is refused unless the
    caller opts in explicitly, and the result is then marked non-authoritative.
    """
    path = Path(path).resolve()
    shown = display_path(path)
    if not path.exists():
        raise SystemExit(
            f"no retrieval result at {shown}.\n"
            "Run the selection stage first:  python evals/retrieval_eval.py")
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"{shown}: not valid JSON ({e}). Re-run evals/retrieval_eval.py.") from None

    if not run.get("complete"):
        raise SystemExit(
            f"{shown}: the retrieval run did not finish (complete=false), so no chunk "
            "configuration was selected. Re-run evals/retrieval_eval.py.")
    if "selected" not in run:
        raise SystemExit(f"{shown}: no selected chunk configuration. Re-run evals/retrieval_eval.py.")
    if not run.get("authoritative") and not allow_smoke:
        raise SystemExit(
            f"{shown}: a smoke selection (stand-in embedder/reranker or --limit), not a "
            "measurement. Pass --allow-smoke-selection for an offline smoke run; the result "
            "is then marked non-authoritative.")
    for key, current, what in (
        ("golden_digest", golden_digest(), "evals/golden_set.jsonl (the gold passages moved)"),
        ("corpus_digest", corpus_digest(), "the corpus manifest or a source document"),
        ("prompt_digest", prompt_digest(), "system_prompt.md"),
    ):
        if run.get(key) != current:
            raise SystemExit(
                f"{shown}: selected with {key} {run.get(key)}, but {what} is now {current}, so "
                "the selection no longer describes this system. Re-run evals/retrieval_eval.py.")
    return run


# Settings a retrieval selection is only valid for. They are taken FROM the
# selection, so a later edit to config.py or .env cannot silently change the
# retrieval being judged. The request budget (context window, completion
# allowance, retries) is frozen too: it decides how many sources the model sees,
# so every candidate must get the same one.
FROZEN_RETRIEVAL_KEYS = (
    "embedder", "embedding_model", "embedding_dim", "reranker", "reranker_model",
    "reranker_max_length", "fuse_rerank_with_dense", "sparse_candidates_k", "governing_candidates_k",
    "final_k", "candidates_k", "rrf_k", "use_dense", "use_sparse", "hnsw_ef_search",
    "balance_sources", "min_government_sources", "min_airline_sources", "chunk_breadcrumb",
    "structured_query", "topic_filter", "max_chunks_per_section", "lane_candidates_k",
    "scope_lanes", "procedural_lanes", "min_airline_sources_legal", "lane_seats",
    "max_lane_seats", "seat_remedy_lanes", "lane_seat_section_diverse",
    "max_seats_per_lane",
    "context_token_budget", "query_instruction", "llm_context_window",
    "context_answer_reserve_tokens", "max_validation_retries",
)
# Selections written before the rename name the answer reserve by its old key.
LEGACY_SELECTION_KEYS = {"llm_max_tokens": "context_answer_reserve_tokens"}


def settings_for_selection(base, selection: dict, args):
    """
    (settings, frozen_differences, cli_overrides) for the selected configuration.

    frozen_differences: {key: (current value, value frozen from the selection)} for
    every setting the current environment would have changed - reported, not hidden.
    cli_overrides: explicit --embedder/--reranker values that differ from the
    selection; they make the run non-authoritative.
    """
    sel = selection["selected"]
    cfg = dict(selection.get("config") or {})
    for old, new in LEGACY_SELECTION_KEYS.items():
        if old in cfg and new not in cfg:
            cfg[new] = cfg.pop(old)
    update = {"chunk_target_tokens": sel["chunk_target_tokens"],
              "chunk_overlap_pct": sel["chunk_overlap_pct"]}
    frozen_differences = {}
    for key in FROZEN_RETRIEVAL_KEYS:
        if key in cfg:
            update[key] = cfg[key]
            if getattr(base, key) != cfg[key]:
                frozen_differences[key] = (getattr(base, key), cfg[key])
    cli_overrides = {}
    for key in ("embedder", "reranker"):
        value = getattr(args, key, None)
        if value and value != cfg.get(key):
            update[key] = value
            cli_overrides[key] = value
    if "embedder" in cli_overrides:
        # The frozen dimension belongs to the selection's embedder, not this one.
        update["embedding_dim"] = 256 if update["embedder"] == "hash" else base.embedding_dim
        update["embedding_model"] = base.embedding_model
    elif update.get("embedder") == "hash" and "embedding_dim" not in cfg:
        update["embedding_dim"] = 256
    from flight_delay.config import revalidate

    return revalidate(base.model_copy(update=update)), frozen_differences, cli_overrides


# --------------------------------------------------------------------------
# Secrets and endpoints
# --------------------------------------------------------------------------


def _dotenv(name: str) -> str | None:
    """A variable from the environment, falling back to a .env line. Never logged."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip("\"'") or None
    return None


def is_local_endpoint(url: str | None) -> bool:
    """True for an endpoint on this machine (a local vLLM): no bill, no shared quota."""
    host = (urlparse(url or "").hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


# --------------------------------------------------------------------------
# Generator candidates
# --------------------------------------------------------------------------


def load_candidates(path: Path = CANDIDATES_FILE) -> dict[str, dict]:
    """
    name -> candidate. A candidate names its provider, endpoint, model and the
    environment variable holding its key (never the key itself).
    """
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for c in data.get("candidates", []):
        missing = [k for k in ("name", "provider", "base_url", "model") if not c.get(k)]
        if missing:
            raise SystemExit(f"{display_path(path)}: a candidate is missing {missing}: {c}")
        if c["name"] in out:
            raise SystemExit(f"{display_path(path)}: duplicate candidate name {c['name']!r}")
        out[c["name"]] = c
    return out


def resolve_generators(specs: list[str], candidates: dict[str, dict], settings) -> list[dict]:
    """
    Each --generator value -> a generator description.

      none        no model (routing and gate calibration only)
      echo        the offline test double
      configured  LLM_* from the environment / .env, as the app would use them
      <name>      a candidate from evals/generator_candidates.json
    """
    specs = specs or ["configured"]
    if "none" in specs and len(specs) > 1:
        raise SystemExit("--generator none runs no model; it cannot be combined with candidates")
    out, seen = [], set()
    for spec in specs:
        if spec in seen:
            raise SystemExit(f"--generator {spec} given twice")
        seen.add(spec)
        if spec in ("none", "echo"):
            out.append({"name": spec, "provider": spec, "model": spec, "base_url": None})
        elif spec == "configured":
            if not (settings.llm_base_url and settings.llm_model):
                raise SystemExit("--generator configured: LLM_BASE_URL/LLM_MODEL are not set (there is no "
                                 "default generator); name a candidate with --generator NAME")
            out.append({"name": f"configured ({settings.llm_model})",
                        "provider": settings.llm_provider, "base_url": settings.llm_base_url,
                        "model": settings.llm_model, "api_key": settings.llm_api_key,
                        "extra_body": settings.llm_extra_body,
                        "max_completion_tokens": settings.llm_max_completion_tokens,
                        "price_input_per_mtok": settings.llm_price_input_per_mtok,
                        "price_output_per_mtok": settings.llm_price_output_per_mtok})
        elif spec in candidates:
            out.append(dict(candidates[spec]))
        else:
            known = ", ".join(sorted(candidates)) or "(no candidates file)"
            raise SystemExit(f"unknown generator {spec!r}: use none, echo, configured or one of: {known}")
    return out


def candidate_api_key(gen: dict) -> tuple[str, str]:
    """
    (key, the variable it came from). A hosted candidate uses ITS OWN `api_key_env`
    variable and nothing else: generator calls never borrow the judge's key, even on
    the same host (the two may hold the same secret, but only because the user set
    both). Reported as a source name, never a value.
    """
    if gen.get("api_key"):
        return gen["api_key"], "LLM_API_KEY (configured)"
    if is_local_endpoint(gen.get("base_url")):
        return "not-needed-for-local", "local endpoint (no key)"
    env = gen.get("api_key_env")
    if env and _dotenv(env):
        return _dotenv(env), env
    raise SystemExit(f"generator {gen['name']}: no API key. Set {env or 'its api_key_env'} "
                     f"in the environment or .env (generator calls never use {JUDGE_ENV_API_KEY}).")


def max_prompt_tokens(settings) -> int:
    """The largest prompt the input assembly may build: budget minus the answer reserve."""
    return settings.llm_context_window - settings.context_answer_reserve_tokens


def generator_settings(settings, gen: dict):
    """The frozen settings with this candidate's provider, endpoint and model."""
    from flight_delay.config import revalidate

    if gen["provider"] in ("none", "echo"):
        return settings.model_copy(update={"llm_provider": "echo", "llm_base_url": "echo",
                                           "allow_test_doubles": True})
    # The frozen llm_context_window is the common INPUT-ASSEMBLY budget (it sizes the
    # sources with context_answer_reserve_tokens held back), not a model window: every
    # candidate sees the same sources. The completion cap (reasoning included) is the
    # candidate's own, and its SERVED window must hold the largest assembled prompt
    # plus that cap. Checked here statically; measured per call in run_cases.
    cap = gen.get("max_completion_tokens")
    provider_cap = gen.get("provider_max_completion_tokens")
    if cap is not None and provider_cap is not None and cap > provider_cap:
        raise SystemExit(f"generator {gen['name']}: max_completion_tokens={cap} exceeds the "
                         f"provider's limit {provider_cap}")
    served = gen.get("context_window")
    reserve = settings.context_answer_reserve_tokens
    needed = max_prompt_tokens(settings) + (cap or reserve)
    if served is not None and served < needed:
        raise SystemExit(
            f"generator {gen['name']}: serves {served} tokens, below the largest assembled prompt "
            f"(input-assembly budget {settings.llm_context_window} - answer reserve {reserve}) + "
            f"completion cap {cap or reserve} = {needed}. Every candidate must accept the same "
            "request, or they would not see the same sources.")
    key, _ = candidate_api_key(gen)
    update = {
        "llm_provider": gen["provider"], "llm_base_url": gen["base_url"],
        "llm_model": gen["model"], "llm_api_key": key,
        "llm_max_completion_tokens": cap,
        "llm_extra_body": gen.get("extra_body") or {},
        "llm_price_input_per_mtok": gen.get("price_input_per_mtok"),
        "llm_price_output_per_mtok": gen.get("price_output_per_mtok"),
        # client pacing for the account's rate limits; not part of the request body
        "llm_min_call_interval_s": gen.get("min_call_interval_s") or 0.0,
    }
    if gen.get("retry_max_wait_s") is not None:
        update["llm_retry_max_wait_s"] = gen["retry_max_wait_s"]
    return revalidate(settings.model_copy(update=update))


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------


def _last_prompt_input(prompt: str) -> dict:
    """The `input: {...}` object a Ragas prompt ends with (used by the offline judge)."""
    marker = prompt.rfind("input: ")
    if marker == -1:
        return {}
    start = prompt.find("{", marker)
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(prompt[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(prompt[start:i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def build_offline_judge():
    """
    A deterministic stand-in for the judge, for smoke runs and tests.

    It implements the same InstructorBaseRagasLLM interface Ragas calls, so the
    real metric pipelines run end to end (statement decomposition, NLI verdicts,
    score arithmetic) with no API key and no model server. Verdicts are decided
    by lexical overlap, which is crude but reproducible — its numbers are a
    smoke signal, never a measurement, and the result JSON records that.
    """
    import re as _re

    from ragas.llms.base import InstructorBaseRagasLLM

    word_re = _re.compile(r"[a-z0-9§$€£]+")

    def sentences(text: str) -> list[str]:
        parts = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text or "") if len(s.strip()) > 15]
        return parts[:12] or ([text.strip()] if text and text.strip() else [])

    def supported(statement: str, context: str) -> int:
        words = {w for w in word_re.findall(statement.lower()) if len(w) > 3}
        if not words:
            return 0
        ctx = set(word_re.findall(context.lower()))
        return int(len(words & ctx) / len(words) >= 0.6)

    class OfflineJudgeLLM(InstructorBaseRagasLLM):
        """Lexical-overlap stand-in; not a language model."""

        model = "offline-lexical-stand-in"

        def generate(self, prompt: str, response_model):
            data = _last_prompt_input(prompt)
            name = response_model.__name__
            if name == "StatementGeneratorOutput":
                return response_model(statements=sentences(data.get("answer", "")))
            if name == "ClaimDecompositionOutput":
                return response_model(claims=sentences(data.get("response", "")))
            if name == "NLIStatementOutput":
                context = data.get("context", "")
                return response_model(statements=[
                    {"statement": s, "reason": "offline lexical-overlap stand-in",
                     "verdict": supported(s, context)}
                    for s in data.get("statements", [])
                ])
            raise TypeError(f"offline judge cannot produce {name}")

        async def agenerate(self, prompt: str, response_model):
            return self.generate(prompt, response_model)

    return OfflineJudgeLLM(), "offline-lexical-stand-in", "offline"


def cap_parameter(base_url: str | None) -> str:
    """The completion-cap field a host documents: Groq deprecates max_tokens."""
    return "max_completion_tokens" if urlparse(base_url or "").hostname == "api.groq.com" else "max_tokens"


class CallMeter:
    """
    Every HTTP response the judge's client receives, as usage records: status,
    prompt/completion/reasoning tokens, finish_reason, truncation. Hooked into the
    httpx client, so instructor re-asks and the SDK's 429 retries are counted too.
    """

    def __init__(self):
        self.calls: list[dict] = []

    async def on_response(self, response) -> None:
        from flight_delay.generation import sanitize_error_text

        await response.aread()
        rec = {"status": response.status_code, "prompt_tokens": None, "completion_tokens": None,
               "reasoning_tokens": None, "finish_reason": None, "truncated": False}
        if response.status_code >= 400:
            # what the provider said, complete and sanitized (keys and account ids removed)
            rec.update(error_code=None, error_type=None, error_message=None,
                       retry_after_s=response.headers.get("retry-after"))
            try:
                error = response.json().get("error")
                if isinstance(error, dict):
                    rec.update(error_code=error.get("code"), error_type=error.get("type"),
                               error_message=sanitize_error_text(error.get("message") or ""))
            except (ValueError, AttributeError, TypeError):
                rec["error_message"] = sanitize_error_text(response.text)
        try:
            data = response.json()
            usage = data.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            rec.update(prompt_tokens=usage.get("prompt_tokens"),
                       completion_tokens=usage.get("completion_tokens"),
                       reasoning_tokens=details.get("reasoning_tokens"))
            choices = data.get("choices") or []
            if choices:
                rec["finish_reason"] = choices[0].get("finish_reason")
                rec["truncated"] = rec["finish_reason"] == "length"
        except (ValueError, AttributeError, TypeError):
            pass
        self.calls.append(rec)


class CallPacer:
    """
    Keeps the start of consecutive judge requests at least `min_interval_s` apart,
    hooked into the httpx client's request event so that instructor's re-asks and
    the OpenAI SDK's own 429 retries are paced too - they are requests like any
    other, and they are exactly what a per-minute limit counts.

    The judge is scored one case at a time, but a single case makes several calls
    back to back (statement decomposition, then verification), which is what
    overruns a requests-per-minute cap. Mirrors LLMClient._pace for the generator.
    """

    def __init__(self, min_interval_s: float = 0.0, clock=None, sleep=None):
        self.min_interval_s = min_interval_s
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._last_start: float | None = None
        self.waited_s = 0.0
        self.waits = 0

    async def on_request(self, request) -> None:
        if self.min_interval_s and self._last_start is not None:
            gap = self.min_interval_s - (self._clock() - self._last_start)
            if gap > 0:
                self.waited_s += gap
                self.waits += 1
                await self._sleep(gap)
        self._last_start = self._clock()


def summarize_calls(calls: list[dict], prices: dict | None) -> dict:
    """Totals over usage records; cost only when both prices are known."""
    prompt = _sum_known(c.get("prompt_tokens") for c in calls)
    completion = _sum_known(c.get("completion_tokens") for c in calls)
    pin, pout = (prices or {}).get("price_input_per_mtok"), (prices or {}).get("price_output_per_mtok")
    return {
        "calls": len(calls),
        "http_errors": sum((c.get("status") or 200) >= 400 for c in calls),
        "prompt_tokens": prompt, "completion_tokens": completion,
        "reasoning_tokens": _sum_known(c.get("reasoning_tokens") for c in calls),
        "truncated_calls": sum(bool(c.get("truncated")) for c in calls),
        "missing_usage_calls": sum(c.get("prompt_tokens") is None for c in calls),
        "cost_usd": (round((prompt or 0) * pin / 1e6 + (completion or 0) * pout / 1e6, 6)
                     if pin is not None and pout is not None and calls else None),
    }


def build_api_judge(base_url: str, model: str, api_key: str, request: dict, max_retries: int = 6,
                    transport=None, min_call_interval_s: float = 0.0):
    """
    The real judge: any OpenAI-compatible endpoint, through Ragas' own adapter.
    Returns (llm, model, base_url, meter).

    AsyncOpenAI, not OpenAI: the collections metrics are async and call
    `agenerate`, which a Ragas LLM built on a synchronous client refuses with
    "Cannot use agenerate() with a synchronous client" — one error per metric per
    case, i.e. a whole run of nothing.

    `request` is sent verbatim with every judge call (JUDGE_REQUEST): the completion
    cap goes out under the host's own field name, well above the adapter's 1024
    default, because a reasoning judge spends tokens thinking before it emits the
    structured object and a truncated object is an IncompleteOutputException, not a
    low score. Ragas' own top_p default is removed so the frozen request is the whole
    request. Structured output is instructor Mode.JSON (JUDGE_STRUCTURED_OUTPUT).
    max_retries: the OpenAI client backs off on 429s itself. `transport`: tests and
    the offline request display only.
    """
    import httpx
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    meter = CallMeter()
    pacer = CallPacer(min_call_interval_s)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0), transport=transport,
                                    event_hooks={"request": [pacer.on_request],
                                                 "response": [meter.on_response]})
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries,
                         http_client=http_client)
    llm = llm_factory(model, provider="openai", client=client)
    llm.model_args.pop("max_tokens", None)
    llm.model_args.pop("top_p", None)
    frozen = dict(request)
    cap = frozen.pop("max_completion_tokens")
    llm.model_args.update(frozen)
    llm.model_args[cap_parameter(base_url)] = cap
    meter.pacer = pacer
    return llm, model, base_url, meter


def judge_endpoint(args, settings) -> tuple[str, str]:
    """(base_url, model) of the judge, without building it."""
    base_url = args.judge_base_url or _dotenv(JUDGE_ENV_BASE_URL) or settings.llm_base_url
    model = args.judge_model or _dotenv(JUDGE_ENV_MODEL) or settings.llm_model
    return base_url, model


def judge_request(args) -> dict:
    """The frozen judge request, with the one CLI override that exists (the cap)."""
    return {**JUDGE_REQUEST, "max_completion_tokens": args.judge_max_tokens}


def judge_min_call_interval_s(base_url: str) -> float:
    """
    Seconds between judge request starts. A local server has no per-minute cap; a
    hosted one does, and a single judged case makes several calls back to back.
    JUDGE_MIN_CALL_INTERVAL_S is sized for AI Studio's free tier (RPM 15).
    """
    return 0.0 if is_local_endpoint(base_url) else JUDGE_MIN_CALL_INTERVAL_S


def judge_api_key(base_url: str) -> str:
    """JUDGE_API_KEY and nothing else for a hosted judge (never the generator's key)."""
    if is_local_endpoint(base_url):
        return _dotenv(JUDGE_ENV_API_KEY) or "not-needed-for-local"
    key = _dotenv(JUDGE_ENV_API_KEY)
    if not key:
        raise SystemExit(f"the judge at {base_url} needs {JUDGE_ENV_API_KEY} (judge calls never use "
                         "a generator key)")
    return key


def build_judge(args, settings):
    """
    (judge_llm, model_name, base_url, meter). Configured independently of the
    generator (JUDGE_*), and ONE judge with ONE frozen request scores every
    candidate in the run.
    """
    if args.judge == "fake":
        return (*build_offline_judge(), None)
    base_url, model = judge_endpoint(args, settings)
    if base_url in ("echo", "none", ""):
        raise SystemExit(
            "no judge endpoint: LLM_BASE_URL is the echo stand-in and JUDGE_BASE_URL is unset. "
            f"Set {JUDGE_ENV_BASE_URL}/{JUDGE_ENV_MODEL}/{JUDGE_ENV_API_KEY}, or use --judge fake "
            "for an offline smoke run.")
    return build_api_judge(base_url, model, judge_api_key(base_url), judge_request(args),
                           min_call_interval_s=judge_min_call_interval_s(base_url))


def preflight_judge(judge_llm, model: str, base_url: str) -> None:
    """
    One tiny structured call to the judge BEFORE anything is indexed or generated.

    Judging runs last, so without this a judge that cannot be reached - wrong model
    id, wrong base url, a key for the wrong provider, an OpenAI-compatibility layer
    that rejects the frozen request - is discovered only after a full generation
    pass has been paid for. On Groq's free tier that pass is most of a day's token
    quota, and it cannot be replayed: the answers live in memory, not on disk.

    This exercises the real path (instructor structured output through the Ragas
    adapter), not just connectivity, because that is where an OpenAI-compatible
    endpoint most often differs. The call is metered like any other.
    """
    from pydantic import BaseModel

    from flight_delay.generation import sanitize_error_text

    class JudgePreflight(BaseModel):
        ok: bool

    try:
        asyncio.run(judge_llm.agenerate('Reply with JSON only: {"ok": true}', JudgePreflight))
    except Exception as e:                                    # noqa: BLE001 - reported, not handled
        raise SystemExit(
            f"the judge {model} at {base_url} failed its preflight call, so nothing was "
            f"generated and no tokens were spent on a candidate:\n"
            f"  {type(e).__name__}: {sanitize_error_text(str(e))}\n"
            f"Check {JUDGE_ENV_MODEL} (the id the provider serves), {JUDGE_ENV_BASE_URL} and "
            f"{JUDGE_ENV_API_KEY} (the judge key is never the generator's), and that the "
            f"endpoint accepts the frozen judge request {JUDGE_REQUEST}.") from e


# --------------------------------------------------------------------------
# Authorized evidence
# --------------------------------------------------------------------------

_RULE_RE = re.compile(r"^RULE (\d+): [^\n]*\n═+\n(.*?)(?=\n═+\nRULE \d+:|\Z)", re.S | re.M)


def prompt_guidance(prompt: str | None = None) -> str:
    """
    The substantive rules of the unchanged system prompt (PROMPT_GUIDANCE_RULES),
    read from system_prompt.md at run time, never copied into code.
    """
    if prompt is None:
        from flight_delay.generation import SYSTEM_PROMPT as prompt
    rules = {int(m.group(1)): m.group(0).strip() for m in _RULE_RE.finditer(prompt)}
    missing = [n for n in PROMPT_GUIDANCE_RULES if n not in rules]
    if missing:
        raise SystemExit(f"system_prompt.md: rule(s) {missing} not found; the judge evidence "
                         "would silently lose the guidance generation was given")
    return "\n\n".join(rules[n] for n in PROMPT_GUIDANCE_RULES)


def source_blocks(built) -> list[str]:
    """
    The numbered source blocks exactly as sent (header line + clause text), split
    from BuiltContext.text rather than rebuilt from chunks.
    """
    if built is None or "SOURCES:\n\n" not in built.text:
        return []
    body = built.text.split("SOURCES:\n\n", 1)[1]
    if not body.strip():
        return []
    blocks = body.split("\n\n---\n\n")
    if len(blocks) != len(built.source_map):
        # A separator inside a clause: keep the exact text as one context rather
        # than guess where a block ends.
        return [body]
    return blocks


def authorized_evidence(answer, guidance: str) -> list[str]:
    """
    Faithfulness contexts for one turn, each labelled with what it is. Empty when
    no generation happened (nothing was authorized because nothing was generated).
    """
    built = answer.built_context
    if built is None:
        return []
    contexts = [f"RETRIEVED SOURCE (sent to the model):\n{b}" for b in source_blocks(built)]
    supplied = list(answer.diagnostics.get("prompt_notes") or [])
    if built.flight_text:
        supplied.append(built.flight_text)
    if supplied:
        contexts.append("SUPPLIED WITH THE QUESTION (route, airline and flight facts; "
                        "not a citable source):\n" + "\n\n".join(supplied))
    contexts.append("SYSTEM PROMPT GUIDANCE (fixed rules given to the model; not a "
                    "retrieved source):\n" + guidance)
    return contexts


# --------------------------------------------------------------------------
# Running cases
# --------------------------------------------------------------------------


class CachingRetriever:
    """
    Retrieval computed once per distinct request and shared by every pass, so all
    candidates answer from identical chunks and generation latency is not mixed
    with embedding and reranking time.
    """

    def __init__(self, inner):
        self.inner = inner
        self.cache: dict[str, object] = {}
        self.misses = 0

    def retrieve(self, query, **kw):
        key = json.dumps([query, kw], sort_keys=True, default=str)
        if key not in self.cache:
            self.misses += 1
            self.cache[key] = self.inner.retrieve(query, **kw)
        return self.cache[key]


class NoGeneration:
    """Pass 0: the turn stops once the context is built. No model is called
    (no `last` attribute, so the pipeline records no model call)."""

    def complete(self, system, user):  # noqa: ARG002
        from flight_delay.generation import LLMError

        class RoutingPassStop(LLMError):
            kind = "routing_pass_no_generation"

        raise RoutingPassStop("routing pass: no generation")


def check_calls(usage: list[dict], served_window: int | None) -> list[dict]:
    """
    Per-call records with the served-window check: prompt_tokens (as the server
    counted them) + the call's completion cap must fit the window the candidate is
    served with. A call without reported prompt tokens cannot be checked and says so.
    """
    out = []
    for u in usage:
        rec = dict(u)
        cap = u.get("max_completion_tokens")
        if served_window is None:
            rec["fits_served_window"] = None
        elif u.get("prompt_tokens") is None or cap is None:
            rec["fits_served_window"] = "unchecked (no prompt_tokens reported)"
        else:
            rec["prompt_plus_cap"] = u["prompt_tokens"] + cap
            rec["fits_served_window"] = rec["prompt_plus_cap"] <= served_window
        out.append(rec)
    return out


def run_cases(pipe, rows, served_window: int | None = None) -> list[dict]:
    """Drive every golden case through the pipeline; collect what each metric needs."""
    from golden.replay import FixtureTool, action_of, replay, run_conversation

    from flight_delay.tools import SUPPORTED_AIRLINES

    guidance = prompt_guidance()
    out = []
    for item in rows:
        action = action_of(item)
        tool = FixtureTool(item.get("flight_fixture"))
        t0 = time.perf_counter()
        ans = run_conversation(pipe, item, tool=tool)
        turn_s = time.perf_counter() - t0

        # The answer above is the FINAL turn, so it cannot show whether production
        # asked first. plan_turn is pure routing, so replaying the same turns
        # costs nothing and says exactly that: did it ask, and did it ask twice.
        first, final, _ = replay(item["question"], item.get("reply"), item.get("flight_fixture"))
        asked_first_turn = first.clarification is not None
        asked_twice = asked_first_turn and final.clarification is not None

        d = ans.diagnostics or {}
        model_answer = d.get("model_answer") or d.get("rejected_answer") or ""
        usage = check_calls(d.get("usage") or [], served_window)
        built = ans.built_context
        out.append({
            "id": item["id"],
            "category": item["category"],
            "tags": list(item.get("tags") or []),
            "action": action,
            "answerable": bool(item.get("answerable")),
            "outcome": ans.outcome,
            "intent": ans.intent,
            "generation_attempted": ans.generation_attempted,
            "user_input": ans.question,
            "response": ans.text,
            "model_answer": model_answer,
            "reference": item.get("expected_output") or "",
            "evidence": authorized_evidence(ans, guidance),
            "sources": {m: f"{c.doc_id}#{c.section_id}"
                        for m, c in (built.source_map.items() if built is not None else [])},
            "context_tokens": built.estimated_tokens if built is not None else 0,
            "gate": d.get("gate"),
            "llm_error": d.get("llm_error_detail"),
            "llm_error_status": d.get("llm_error_status"),
            "llm_error_code": d.get("llm_error_code"),
            "llm_error_retry_after_s": d.get("llm_error_retry_after_s"),
            # The model's final output names only supplied sources. A shown answer
            # passed validation, so only a rejected one can fail here.
            "citation_valid": not d.get("unknown_markers"),
            "citation_coverage": d.get("citation_coverage"),
            "n_citations": len(ans.citations),
            "retry_count": ans.retry_count,
            "validation_failures": ans.validation_failures,
            "clarified": ans.outcome == "clarified" or asked_first_turn,
            "asked_twice": asked_twice,
            "assumption": ans.assumption,
            "unsupported_airline": ans.unsupported_airline,
            "expects_unsupported": item.get("unsupported_airline") is not None,
            "flight_lookups": list(tool.calls),
            "supported_airline_sources": sorted(
                {c.airline_iata for c in ans.context_chunks if c.airline_iata in SUPPORTED_AIRLINES}),
            "requires_flight": item.get("flight_fixture") is not None,
            "got_flight_data": ans.live_data is not None,
            "prompt_tokens": _sum_known(u.get("prompt_tokens") for u in usage),
            "completion_tokens": _sum_known(u.get("completion_tokens") for u in usage),
            "reasoning_tokens": _sum_known(u.get("reasoning_tokens") for u in usage),
            "finish_reasons": [u.get("finish_reason") for u in usage],
            "llm_call_records": usage,
            "llm_calls": len(usage),
            "turn_s": round(turn_s, 3),
        })
    return out


def _sum_known(values) -> int | None:
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


# --------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------


def judge_plan(rows: list[dict]) -> list[dict]:
    """
    For each case, what the judge is asked to score, and the basis of every value
    that is NOT judged. Nothing is dropped: an answer-required case with no
    delivered answer gets factual_correctness 0.0 by definition.
    """
    plan = []
    for r in rows:
        shown = r["outcome"] in SHOWN_GENERATED and r["response"].strip() and r["evidence"]
        p = {"id": r["id"], "judge_faithfulness": bool(shown), "judge_factual": False,
             "factual_basis": "not applicable (no reference answer: action=" + r["action"] + ")"}
        if r["action"] == "answer":
            if not r["reference"].strip():
                p["factual_basis"] = "not applicable (empty reference)"
            elif shown and r["model_answer"].strip():
                p["judge_factual"] = True
                p["factual_basis"] = "judged"
            else:
                p["factual_basis"] = f"no answer delivered (outcome={r['outcome']}): scored 0.0"
        p["faithfulness_basis"] = ("judged" if shown else
                                   f"not applicable (outcome={r['outcome']}: no generated answer shown)")
        plan.append(p)
    return plan


async def _score(rows: list[dict], plan: list[dict], judge_llm, progress=None, meter=None) -> None:
    """
    Faithfulness and FactualCorrectness, one case at a time, results written into
    each row's "ragas". Sequential, because a hosted judge has a per-minute token
    limit and a local one is a single server; per case, so one failed judgement is
    recorded against its case instead of losing the run.
    """
    from ragas.metrics.collections import FactualCorrectness, Faithfulness

    from flight_delay.generation import sanitize_error_text

    faithfulness = Faithfulness(llm=judge_llm)
    factual = FactualCorrectness(llm=judge_llm)

    def value(result):
        v = float(result.value)
        return None if math.isnan(v) else v

    for i, (r, p) in enumerate(zip(rows, plan, strict=True), start=1):
        scores: dict = {"faithfulness": None, "faithfulness_basis": p["faithfulness_basis"],
                        "factual_correctness": None, "factual_basis": p["factual_basis"],
                        "errors": []}
        first_call = len(meter.calls) if meter is not None else 0
        if p["factual_basis"].startswith("no answer delivered"):
            scores["factual_correctness"] = 0.0
        if p["judge_faithfulness"]:
            try:
                scores["faithfulness"] = value(await faithfulness.ascore(
                    user_input=r["user_input"], response=r["response"],
                    retrieved_contexts=r["evidence"]))
                if scores["faithfulness"] is None:
                    scores["faithfulness_basis"] = "judged: no checkable statements (NaN)"
            except Exception as e:
                scores["errors"].append(sanitize_error_text(f"faithfulness: {type(e).__name__}: {e}"))
        if p["judge_factual"]:
            try:
                scores["factual_correctness"] = value(await factual.ascore(
                    response=r["model_answer"], reference=r["reference"]))
            except Exception as e:
                scores["errors"].append(sanitize_error_text(f"factual_correctness: {type(e).__name__}: {e}"))
        # Every HTTP call the judge made for this case (re-asks and 429 retries included).
        r["judge_calls"] = list(meter.calls[first_call:]) if meter is not None else []
        r["ragas"] = scores
        if progress:
            progress(i, len(rows))


def to_ragas_dataset(rows: list[dict]):
    """
    The judged rows in the current Ragas schema, in memory (no second golden CSV).

        user_input         final effective question after replay/clarification
        retrieved_contexts the authorized evidence (see EVIDENCE_SCOPE)
        response           the answer as shown
        reference          the golden reference answer
    """
    from ragas import EvaluationDataset, SingleTurnSample

    return EvaluationDataset(samples=[
        SingleTurnSample(user_input=r["user_input"], retrieved_contexts=list(r["evidence"]),
                         response=r["response"], reference=r["reference"])
        for r in rows
    ])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def deterministic_metrics(rows: list[dict]) -> dict:
    """
    Safety and outcome metrics from Answer.outcome, each with its denominator and
    its failing case ids. A rate without a denominator is unreadable.
    """
    def metric(name, subset, ok):
        failed = [r["id"] for r in subset if not ok(r)]
        return {name: _ratio(len(subset) - len(failed), len(subset)),
                f"{name}_n": len(subset), f"{name}_failed": failed}

    attempted = [r for r in rows if r["generation_attempted"]]
    with_output = [r for r in attempted if r["outcome"] != "llm_error"]
    abstainers = [r for r in rows if r["action"] == "abstain"]
    clarifiers = [r for r in rows if r["action"] == "clarify"]
    unsupported = [r for r in rows if r["expects_unsupported"]]
    answer_cases = [r for r in rows if r["action"] == "answer"]

    out: dict = {"outcomes": dict(sorted(
        {o: sum(r["outcome"] == o for r in rows) for o in {r["outcome"] for r in rows}}.items()))}
    out |= metric("citation_validity", with_output, lambda r: r["citation_valid"])
    out |= metric("validation_pass_rate", with_output, lambda r: r["outcome"] in SHOWN_GENERATED)
    out |= metric("first_try_pass_rate", with_output,
                  lambda r: r["outcome"] in SHOWN_GENERATED and r["retry_count"] == 0)
    llm_errors = [r["id"] for r in attempted if r["outcome"] == "llm_error"]
    out["llm_error_rate"] = _ratio(len(llm_errors), len(attempted))
    out["llm_error_rate_n"] = len(attempted)
    out["llm_error_cases"] = llm_errors
    out["citation_coverage"] = _mean(r["citation_coverage"] for r in with_output)
    # Declining is CORRECT here: out-of-scope questions and uncovered carriers.
    out |= metric("abstention_accuracy", abstainers, lambda r: r["outcome"] in DECLINED)
    # Asking is CORRECT here, once: the missing detail decides which regime governs.
    out |= metric("clarification_accuracy", clarifiers,
                  lambda r: r["clarified"] and not r["asked_twice"])
    def unsupported_ok(r):
        return (not r["flight_lookups"] and not r["supported_airline_sources"]
                and r["unsupported_airline"] is not None)

    out |= metric("unsupported_airline_accuracy", unsupported, unsupported_ok)
    # Hard gate: no incorrect answer on an unsupported-airline or out-of-scope case.
    # An abstain case must be declined; a carrier case must be handled as unsupported.
    scoped = [r for r in rows if {"unsupported_airline", "out_of_scope"} & set(r.get("tags") or ())
              or r["expects_unsupported"]]
    out |= metric("scope_safety", scoped,
                  lambda r: ((r["action"] != "abstain" or r["outcome"] in DECLINED)
                             and (not r["expects_unsupported"] or unsupported_ok(r))
                             and r["outcome"] not in ("llm_error", "validation_failed")))
    # The opposite failure: an answer-required question declined (gate or model).
    over = [r["id"] for r in answer_cases if r["outcome"] in ("gated", "abstained")]
    out["over_abstention_rate"] = _ratio(len(over), len(answer_cases))
    out["over_abstention_rate_n"] = len(answer_cases)
    out["over_abstention_cases"] = over
    # Documented routing gap: production asks where the golden set says it need not.
    out["answer_cases_production_asked"] = [
        r["id"] for r in answer_cases if r["clarified"]]
    out["hybrid_missing_flight_data"] = [
        r["id"] for r in rows if r["requires_flight"] and not r["got_flight_data"]]
    return out


def judge_usage_summary(rows: list[dict], prices: dict | None) -> dict:
    """The judge's calls, tokens and cost while scoring this candidate."""
    judged = [r for r in rows if r.get("judge_calls")]
    s = summarize_calls([c for r in rows for c in r.get("judge_calls", [])], prices)
    return {f"judge_{k}": v for k, v in s.items()} | {
        "judge_cases_with_calls": len(judged),
        "judge_cost_per_judged_case_usd": (round(s["cost_usd"] / len(judged), 6)
                                           if s["cost_usd"] is not None and judged else None)}


def ragas_summary(rows: list[dict]) -> dict:
    """Both judged metrics with explicit denominators, failures and judge errors."""
    judged = [r for r in rows if "ragas" in r]
    faith = [r for r in judged if r["ragas"]["faithfulness_basis"].startswith("judged")]
    answer = [r for r in judged if r["action"] == "answer"
              and not r["ragas"]["factual_basis"].startswith("not applicable")]
    no_answer = [r["id"] for r in answer if r["ragas"]["factual_basis"].startswith("no answer")]
    errors = {r["id"]: r["ragas"]["errors"] for r in judged if r["ragas"]["errors"]}
    fc_scored = [r for r in answer if r["ragas"]["factual_correctness"] is not None]
    fc_judged = [r for r in fc_scored if r["ragas"]["factual_basis"] == "judged"]
    return {
        "evidence_scope": EVIDENCE_SCOPE,
        "faithfulness": _mean(r["ragas"]["faithfulness"] for r in faith),
        "faithfulness_n": sum(r["ragas"]["faithfulness"] is not None for r in faith),
        "faithfulness_applicable_n": len(faith),
        "faithfulness_no_statements": [r["id"] for r in faith
                                       if r["ragas"]["faithfulness"] is None and not r["ragas"]["errors"]],
        "factual_correctness": _mean(r["ragas"]["factual_correctness"] for r in fc_scored),
        "factual_correctness_n": len(fc_scored),
        "factual_correctness_required_n": len(answer),
        "factual_correctness_no_answer_cases": no_answer,
        "factual_correctness_judged_only": _mean(r["ragas"]["factual_correctness"] for r in fc_judged),
        "factual_correctness_judged_only_n": len(fc_judged),
        "judge_error_cases": errors,
        "judge_errors": sum(len(v) for v in errors.values()),
        "judged_calls_planned": len(faith) + len([r for r in answer
                                                  if r["ragas"]["factual_basis"] == "judged"]),
    }


def usage_summary(rows: list[dict], gen: dict) -> dict:
    attempted = [r for r in rows if r["generation_attempted"]]
    prompt = [r["prompt_tokens"] for r in attempted if r["prompt_tokens"] is not None]
    completion = [r["completion_tokens"] for r in attempted if r["completion_tokens"] is not None]
    calls = [c for r in attempted for c in r.get("llm_call_records", [])]
    out = {
        "generation_cases": len(attempted),
        "llm_calls": sum(r["llm_calls"] for r in attempted),
        "usage_reported_cases": len(prompt),
        "prompt_tokens_total": sum(prompt) if prompt else None,
        "prompt_tokens_max": max((c["prompt_tokens"] for c in calls if c.get("prompt_tokens") is not None),
                                 default=None),
        "completion_tokens_total": sum(completion) if completion else None,
        # Reasoning is part of completion_tokens; None when the provider does not report it.
        "reasoning_tokens_total": _sum_known(r["reasoning_tokens"] for r in attempted),
        "reasoning_chars_total": _sum_known(c.get("reasoning_chars") for c in calls),
        "completion_cap": gen.get("max_completion_tokens"),
        "served_window": gen.get("context_window"),
        "truncated_calls": sum(bool(c.get("truncated")) for c in calls),
        "failed_calls": sum(bool(c.get("error")) for c in calls),
        "missing_usage_calls": sum(c.get("prompt_tokens") is None for c in calls),
        "served_window_violations": [
            {"id": r["id"], "prompt_tokens": c["prompt_tokens"], "prompt_plus_cap": c["prompt_plus_cap"]}
            for r in attempted for c in r.get("llm_call_records", []) if c.get("fits_served_window") is False],
        "served_window_unchecked_calls": sum(isinstance(c.get("fits_served_window"), str) for c in calls),
        "tokens_per_case": (round((sum(prompt) + sum(completion)) / len(prompt), 1)
                            if prompt and completion else None),
        "generation_s_mean": _mean(r["turn_s"] for r in attempted),
        "generation_s_p90": (sorted(r["turn_s"] for r in attempted)[int(0.9 * (len(attempted) - 1))]
                             if attempted else None),
        "context_tokens_mean": _mean(r["context_tokens"] for r in attempted),
    }
    pin, pout = gen.get("price_input_per_mtok"), gen.get("price_output_per_mtok")
    # A local endpoint sends no bill: its API cost is measured as 0 and labelled so.
    # A hosted one without prices has an UNKNOWN cost, which is not a cost of zero.
    if is_local_endpoint(gen.get("base_url")):
        out["cost_usd"], out["cost_basis"] = 0.0, "local endpoint: no API charge (hardware not costed)"
    elif pin is not None and pout is not None and prompt and completion:
        out["cost_usd"] = round(sum(prompt) * pin / 1e6 + sum(completion) * pout / 1e6, 6)
        out["cost_basis"] = f"measured tokens x ${pin}/${pout} per M"
    else:
        out["cost_usd"], out["cost_basis"] = None, "unknown (no prices or no usage)"
    out["cost_per_case_usd"] = (round(out["cost_usd"] / len(attempted), 6)
                                if out["cost_usd"] is not None and attempted else None)
    return out


# --------------------------------------------------------------------------
# Gate calibration
# --------------------------------------------------------------------------

GATE_GRID = {
    "min_rerank_score": [None, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0],
    "min_score_margin": [0.0, 0.25, 0.5, 1.0],
    "min_grounding_ratio": [0.0, 0.1, 0.2, 0.25, 0.3, 0.34],
}


def gate_would_answer(signals: dict, floor, margin, ratio) -> bool:
    """ConfidenceGate.evaluate's rules, applied to recorded signals."""
    if not signals.get("n_chunks"):
        return False
    top = signals.get("rerank_top")
    if floor is not None and top is not None and not math.isnan(top) and top < floor:
        return False
    if "score_margin" in signals and signals["score_margin"] < margin:
        return False
    return not ("grounding_ratio" in signals and signals["grounding_ratio"] < ratio)


def gate_calibration(rows: list[dict], settings) -> dict:
    """
    What the gate did on every case that reached it, and what a grid of thresholds
    would have done. Calibration data only: config.py is never changed here.
    """
    reached = [r for r in rows if r.get("gate")]
    answer = [r for r in reached if r["action"] == "answer"]
    abstain = [r for r in reached if r["action"] == "abstain"]
    current = (settings.min_rerank_score, settings.min_score_margin, settings.min_grounding_ratio)

    grid = []
    for floor in GATE_GRID["min_rerank_score"]:
        for margin in GATE_GRID["min_score_margin"]:
            for ratio in GATE_GRID["min_grounding_ratio"]:
                grid.append({
                    "min_rerank_score": floor, "min_score_margin": margin,
                    "min_grounding_ratio": ratio,
                    "false_abstentions": sum(not gate_would_answer(r["gate"]["signals"], floor, margin, ratio)
                                             for r in answer),
                    "missed_abstentions": sum(gate_would_answer(r["gate"]["signals"], floor, margin, ratio)
                                              for r in abstain),
                })
    # Pareto front: no other setting is at least as good on both counts and better on one.
    front = [g for g in grid if not any(
        (o["false_abstentions"] <= g["false_abstentions"] and o["missed_abstentions"] <= g["missed_abstentions"]
         and (o["false_abstentions"], o["missed_abstentions"]) != (g["false_abstentions"], g["missed_abstentions"]))
        for o in grid)]
    seen, pareto = set(), []
    for g in sorted(front, key=lambda g: (g["missed_abstentions"], g["false_abstentions"])):
        key = (g["false_abstentions"], g["missed_abstentions"])
        if key not in seen:
            seen.add(key)
            pareto.append(g)

    def case(r):
        s = r["gate"]["signals"]
        return {"id": r["id"], "action": r["action"], "gate_answers": r["gate"]["should_answer"],
                "rerank_top": s.get("rerank_top"), "score_margin": s.get("score_margin"),
                "grounding_ratio": s.get("grounding_ratio"), "reasons": r["gate"]["reasons"]}

    return {
        "current_thresholds": dict(zip(GATE_GRID, current, strict=True)),
        "cases_reaching_gate": len(reached),
        "answer_cases_reaching_gate": len(answer),
        "abstain_cases_reaching_gate": len(abstain),
        "abstain_cases_not_reaching_gate": [r["id"] for r in rows
                                            if r["action"] == "abstain" and not r.get("gate")],
        "false_abstentions": [r["id"] for r in answer if not r["gate"]["should_answer"]],
        "missed_abstentions": [r["id"] for r in abstain if r["gate"]["should_answer"]],
        "pareto_front": pareto,
        "cases": [case(r) for r in reached],
        "caveat": (f"only {len(abstain)} abstain case(s) reach the gate, so a missed-abstention "
                   "count rests on very little evidence; do not tune to zero false abstentions "
                   "on this set alone"),
    }


# --------------------------------------------------------------------------
# Bake-off recommendation
# --------------------------------------------------------------------------


def hard_gate_failures(s: dict) -> list[str]:
    """Every hard gate a candidate's summary fails (empty = eligible for ranking)."""
    out = []
    if not s.get("generation_cases"):
        out.append("no generation ran")
    if s.get("llm_error_rate") is None or s["llm_error_rate"] > MAX_LLM_ERROR_RATE:
        out.append(f"generation error rate {s.get('llm_error_rate')} > {MAX_LLM_ERROR_RATE}")
    planned = s.get("judged_calls_planned") or 0
    judge_share = (s.get("judge_errors") or 0) / planned if planned else None
    if judge_share is None or judge_share > MAX_JUDGE_ERROR_SHARE:
        out.append(f"judge error share {judge_share} > {MAX_JUDGE_ERROR_SHARE} (after retries)")
    if s.get("citation_validity_failed"):
        out.append(f"fabricated/unknown citations on {', '.join(s['citation_validity_failed'])}")
    if s.get("scope_safety_failed"):
        out.append("incorrect answer on unsupported-airline/out-of-scope cases "
                   f"{', '.join(s['scope_safety_failed'])}")
    for name in ("clarification_accuracy", "abstention_accuracy"):
        if not s.get(f"{name}_n"):
            out.append(f"{name}: no preserved edge case was evaluated")
        elif s.get(f"{name}_failed"):
            out.append(f"{name} failed on {', '.join(s[name + '_failed'])}")
    if s.get("faithfulness") is None or s.get("factual_correctness") is None:
        out.append("no judged Faithfulness/FactualCorrectness score")
    return out


def production_settings(gen: dict, settings) -> dict:
    """
    The explicit production configuration for a candidate, as environment
    variables: its tested completion cap is stated, never left to fall back to the
    answer reserve, and the input-assembly budget stays the one it was tested with.

    min_call_interval_s is deliberately NOT among them (Session 19). It is an
    evaluation-harness control: the harness fires every golden case back to back, so
    on a rate-limited free tier each request collides with the one before it. A
    served app answers one question at a time, and a burst there is already handled
    by LLM_MAX_RETRIES with Retry-After. Carrying the pacing into production would
    instead make a validation retry sit out the full interval before the user got an
    answer. The value used for the measurement is reported in the run plan and kept
    in evals/generator_candidates.json.
    """
    env = {
        "LLM_PROVIDER": gen["provider"], "LLM_BASE_URL": gen["base_url"], "LLM_MODEL": gen["model"],
        "LLM_API_KEY": f"<value of {gen.get('api_key_env') or 'no key: local'}>",
        "LLM_EXTRA_BODY": json.dumps(gen.get("extra_body") or {}),
        "LLM_MAX_COMPLETION_TOKENS": gen.get("max_completion_tokens"),
        "CONTEXT_ANSWER_RESERVE_TOKENS": settings.context_answer_reserve_tokens,
        "LLM_CONTEXT_WINDOW": settings.llm_context_window,
        "LLM_TEMPERATURE": settings.llm_temperature,
    }
    if gen.get("price_input_per_mtok") is not None:
        env["LLM_PRICE_INPUT_PER_MTOK"] = gen["price_input_per_mtok"]
        env["LLM_PRICE_OUTPUT_PER_MTOK"] = gen["price_output_per_mtok"]
    if gen.get("retry_max_wait_s") is not None:
        env["LLM_RETRY_MAX_WAIT_S"] = gen["retry_max_wait_s"]
    return env


def selection_basis(rows: list[dict], all_rows: list[dict], args) -> str:
    """
    One line naming the evidence a recommendation rests on: how the cases were
    chosen, and how many cases each hard gate is actually counted over. A gate with
    no cases in it passes vacuously, so the denominators belong next to the verdict.
    """
    if len(rows) == len(all_rows):
        return f"the full canonical golden set ({len(all_rows)} cases)"
    if getattr(args, "pilot", False):
        how = "--pilot (two fixed wiring cases)"
    elif getattr(args, "sample", None):
        how = f"--sample {args.sample}, stratified by category, seed {args.sample_seed}"
    elif getattr(args, "cases", None):
        how = "--cases (hand-picked ids)"
    else:
        how = f"--limit {args.limit} (the FIRST {args.limit} in golden order, not representative)"
    actions: dict[str, int] = {}
    for r in rows:
        behavior = r.get("expected_behavior") or {}
        action = behavior.get("action") if isinstance(behavior, dict) else None
        actions[action or "unknown"] = actions.get(action or "unknown", 0) + 1
    spread = ", ".join(f"{n} {a}" for a, n in sorted(actions.items()))
    return (f"a SUBSET of {len(rows)} of {len(all_rows)} golden cases via {how} -> {spread}. "
            "Provisional: the hard gates are counted over these cases only, so a gate with one "
            "case in it turns on a single answer, and the ranking metrics carry wide error bars.")


def recommend_generator(candidates: list[dict], settings=None, full_run: bool = True,
                        *, wiring_only: bool = False, basis: str = "") -> dict:
    """
    SELECTION_RULE, printed and never applied. `candidates` are {"name", "summary"}
    (and "generator") dicts with deterministic, ragas and usage summaries merged.

    Session 19 (user's decision): a SUBSET run may now produce a recommendation, so
    that a generator can be adopted without the full 50-case bake-off, which Groq's
    free tier cannot complete (it needs ~493,000 generator tokens against a 200,000
    tokens-per-day cap). Every returned dict carries `basis`, naming how many cases
    the recommendation rests on, and the run itself stays non-authoritative. A
    recommendation from 10 cases is a provisional decision, not a measurement of the
    production system - the hard gates in particular are counted over very few cases
    (with 10 stratified cases, typically one clarify and one abstain), so one flake
    flips them.

    `--pilot` still selects nothing: it is two fixed cases for checking wiring
    (user's rule, Session 14) and was never meant to carry a decision.
    """
    if not candidates:
        return {"recommended": None, "reason": "no candidates", "rule": SELECTION_RULE,
                "basis": basis}
    if wiring_only:
        return {"recommended": None, "rule": SELECTION_RULE, "basis": basis,
                "reason": "--pilot: wiring only, never used to choose"}
    ineligible = {c["name"]: f for c in candidates if (f := hard_gate_failures(c["summary"]))}
    eligible = [c for c in candidates if c["name"] not in ineligible]
    if not eligible:
        return {"recommended": None, "ineligible": ineligible, "rule": SELECTION_RULE,
                "basis": basis, "reason": "no candidate passes the hard gates: no winner"}
    best_faith = max(c["summary"]["faithfulness"] for c in eligible)
    kept = [c for c in eligible if c["summary"]["faithfulness"] >= best_faith - FAITHFULNESS_TOLERANCE]
    inf = float("inf")

    def known(v):
        return inf if v is None else v

    pick = min(kept, key=lambda c: (-c["summary"]["factual_correctness"],
                                    known(c["summary"].get("cost_per_case_usd")),
                                    known(c["summary"].get("tokens_per_case")),
                                    known(c["summary"].get("generation_s_mean"))))
    out = {"recommended": pick["name"], "ineligible": ineligible, "rule": SELECTION_RULE,
           "basis": basis, "full_run": full_run,
           "within_faithfulness_tolerance": [c["name"] for c in kept]}
    if settings is not None and pick.get("generator"):
        out["production_settings"] = production_settings(pick["generator"], settings)
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

TABLE_COLUMNS = [
    ("generator", "name", "{}"),
    ("faithful", "faithfulness", "{:.3f}"),
    ("n", "faithfulness_n", "{}"),
    ("factual", "factual_correctness", "{:.3f}"),
    ("n", "factual_correctness_n", "{}"),
    ("judged-only", "factual_correctness_judged_only", "{:.3f}"),
    ("valid", "validation_pass_rate", "{:.2f}"),
    ("1st try", "first_try_pass_rate", "{:.2f}"),
    ("llm err", "llm_error_rate", "{:.2f}"),
    ("over-abst", "over_abstention_rate", "{:.2f}"),
    ("tok/case", "tokens_per_case", "{:.0f}"),
    ("gen s", "generation_s_mean", "{:.1f}"),
    ("cost $", "cost_usd", "{:.4f}"),
    ("$/case", "cost_per_case_usd", "{:.5f}"),
    ("judge $", "judge_cost_usd", "{:.4f}"),
]


def _table(rows: list[dict], columns) -> str:
    header = [c[0] for c in columns]
    body = [[("n/a" if r.get(k) is None else fmt.format(r[k])) for _, k, fmt in columns] for r in rows]
    widths = [max(len(header[i]), *(len(b[i]) for b in body)) if body else len(header[i])
              for i in range(len(header))]
    lines = ["  ".join(h.rjust(w) for h, w in zip(header, widths, strict=True)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.rjust(w) for c, w in zip(b, widths, strict=True)) for b in body]
    return "\n".join(lines)


def print_report(run: dict) -> None:
    sel = run["selected"]
    print("\n" + "=" * 78)
    print("GENERATION EVALUATION")
    print("=" * 78)
    print(f"retrieval: {sel['chunk_target_tokens']} tokens / {sel['chunk_overlap_pct']}% overlap, "
          f"backend {run['config']['backend']}, {run['config']['n_chunks']} chunks, "
          f"golden v{run['golden_version']}, {run['n_cases']} cases")
    judge = run["judge"]
    if judge:
        print(f"judge (frozen for every candidate): {judge['model']} @ {judge['base_url']}")
        if judge["is_offline_stand_in"]:
            print("  SMOKE TEST: the offline lexical stand-in is not a language model.")
        if judge["self_evaluation"]:
            print(f"  WARNING: the judge is also a candidate ({', '.join(judge['self_evaluation'])}); "
                  "its own scores are an upper bound.")
        print(f"evidence scope: {EVIDENCE_SCOPE}")
        print("  WARNING: every reference answer is an UNVETTED DRAFT; factual_correctness is a "
              "consistency signal, not validated accuracy.")

    g = run["gate_calibration"]
    off = run["config"].get("gate_for_candidates") == "off"
    print("\nCONFIDENCE GATE as configured (routing pass; "
          + ("candidates ran with the gate OFF" if off else "the same gate applies to every candidate")
          + f"; thresholds {g['current_thresholds']})")
    print(f"  reached the gate: {g['cases_reaching_gate']} cases "
          f"({g['answer_cases_reaching_gate']} answer, {g['abstain_cases_reaching_gate']} abstain)")
    print(f"  false abstentions (answer cases refused): {len(g['false_abstentions'])} "
          f"{', '.join(g['false_abstentions'])}")
    print(f"  missed abstentions (abstain cases let through): {len(g['missed_abstentions'])} "
          f"{', '.join(g['missed_abstentions'])}")
    print("  threshold grid, Pareto front (false abstentions / missed abstentions):")
    for p in g["pareto_front"][:8]:
        print(f"    rerank>={p['min_rerank_score']}  margin>={p['min_score_margin']}  "
              f"grounding>={p['min_grounding_ratio']}  ->  {p['false_abstentions']} / "
              f"{p['missed_abstentions']}")
    print(f"  {g['caveat']}")

    if not run["candidates"]:
        print("\nno generator ran (--generator none): routing and gate data only.")
        return
    print("\nCANDIDATES")
    print(_table([{"name": c["name"], **c["summary"]} for c in run["candidates"]], TABLE_COLUMNS))
    for c in run["candidates"]:
        s = c["summary"]
        print(f"\n  {c['name']}  ({c['generator']['model']} @ {c['generator']['base_url']})")
        print(f"    outcomes {s['outcomes']}")
        print(f"    faithfulness n={s['faithfulness_n']} of {s['faithfulness_applicable_n']} shown "
              f"answers; factual_correctness n={s['factual_correctness_n']} of "
              f"{s['factual_correctness_required_n']} answer-required "
              f"(no answer delivered, scored 0: {len(s['factual_correctness_no_answer_cases'])})")
        if s["judge_errors"]:
            print(f"    JUDGE ERRORS {s['judge_errors']} on {', '.join(s['judge_error_cases'])} "
                  "(excluded from the means)")
        for name in ("abstention_accuracy", "clarification_accuracy", "unsupported_airline_accuracy",
                     "citation_validity"):
            print(f"    {name} {s[name]} (n={s[name + '_n']})"
                  + (f" FAILED {', '.join(s[name + '_failed'])}" if s[name + "_failed"] else ""))
        print(f"    scope_safety {s['scope_safety']} (n={s['scope_safety_n']})"
              + (f" FAILED {', '.join(s['scope_safety_failed'])}" if s["scope_safety_failed"] else ""))
        print(f"    generator calls {s['llm_calls']} (failed {s['failed_calls']}, truncated "
              f"{s['truncated_calls']}, no usage {s['missing_usage_calls']}); prompt tokens "
              f"{s['prompt_tokens_total']} (max {s['prompt_tokens_max']}), completion "
              f"{s['completion_tokens_total']}, reasoning tokens {s['reasoning_tokens_total']} "
              f"(reasoning chars {s['reasoning_chars_total']}); cap {s['completion_cap']}, served "
              f"window {s['served_window']}")
        if s["served_window_violations"] or s["served_window_unchecked_calls"]:
            print(f"    SERVED WINDOW: prompt + cap exceeded it on {s['served_window_violations']}; "
                  f"unchecked calls {s['served_window_unchecked_calls']}")
        print(f"    generator cost ${s['cost_usd']} ({s['cost_basis']}); per case ${s['cost_per_case_usd']}")
        if "judge_calls" in s:
            print(f"    judge calls {s['judge_calls']} (HTTP errors {s['judge_http_errors']}, truncated "
                  f"{s['judge_truncated_calls']}); prompt {s['judge_prompt_tokens']}, completion "
                  f"{s['judge_completion_tokens']}, reasoning {s['judge_reasoning_tokens']}; cost "
                  f"${s['judge_cost_usd']} (per judged case ${s['judge_cost_per_judged_case_usd']})")
        if s["llm_error_cases"]:
            print(f"    llm errors: {', '.join(s['llm_error_cases'])}")
        if s["hybrid_missing_flight_data"]:
            print(f"    WARNING hybrid cases without their flight record: "
                  f"{', '.join(s['hybrid_missing_flight_data'])}")
    first = run["candidates"][0]["summary"]
    if first["answer_cases_production_asked"]:
        print("\n  production asked on cases the golden set says are answerable as asked: "
              f"{', '.join(first['answer_cases_production_asked'])}")
    est = run.get("full_run_estimate")
    if est:
        print(f"\nFULL-RUN COST ESTIMATE: ${est['total_usd']}  ({est['basis']})")
        for name, p in est["per_candidate"].items():
            print(f"  {name}: generator ${p['generator_usd']} + judge ${p['judge_usd']} = ${p['total_usd']}")
    rec = run["recommendation"]
    print(f"\nRECOMMENDED GENERATOR (not applied): {rec.get('recommended')}"
          + (f"  ({rec['reason']})" if rec.get("reason") else ""))
    if rec.get("basis"):
        print(f"  basis: {rec['basis']}")
    if rec.get("recommended") and not rec.get("full_run", True):
        print("  PROVISIONAL: this recommendation comes from a subset, not the full golden set. "
              "Adopting it is a decision to proceed, not a measurement of the production system; "
              "record it as provisional wherever the generator is configured.")
    print(f"  rule: {SELECTION_RULE}")
    for name, why in (rec.get("ineligible") or {}).items():
        print(f"  ineligible {name}: {'; '.join(why) if isinstance(why, list) else why}")
    if rec.get("production_settings"):
        print("  production settings to adopt (explicit, including the tested completion cap):")
        for k, v in rec["production_settings"].items():
            print(f"    {k}={v}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def run_is_authoritative(selection: dict, *, cli_overrides: dict, backend: str, limit, gate: str,
                         judge: str, self_judged: list, generators: list[dict]) -> bool:
    """
    A measurement of the production system: an authoritative selection, the
    production store, every case, the configured gate, a real separate judge and
    real generators. Anything else writes latest-smoke.json.
    """
    return (bool(selection.get("authoritative")) and not cli_overrides and backend == "postgres"
            and not limit and gate == "configured" and judge == "api" and not self_judged
            and bool(generators)
            and all(g["provider"] not in ("echo", "none") for g in generators))


def select_rows(rows: list[dict], args) -> list[dict]:
    """
    The cases candidates run on: every case, --pilot (PILOT_CASE_IDS), --cases (fixed
    ids, in golden order), --sample (a seeded stratified draw) or --limit (first N,
    smoke only). Unknown ids are refused.
    """
    chosen = [a for a in (getattr(args, "pilot", False), getattr(args, "cases", None),
                          getattr(args, "sample", None), getattr(args, "limit", None)) if a]
    if len(chosen) > 1:
        raise SystemExit("use only one of --pilot, --cases, --sample and --limit")
    ids = PILOT_CASE_IDS if getattr(args, "pilot", False) else (
        [i.strip() for i in args.cases.split(",") if i.strip()] if getattr(args, "cases", None) else None)
    if ids is not None:
        known = {r["id"] for r in rows}
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise SystemExit(f"unknown golden case ids: {', '.join(unknown)}")
        return [r for r in rows if r["id"] in set(ids)]
    if getattr(args, "sample", None):
        if args.sample > len(rows):
            raise SystemExit(f"--sample {args.sample} is more than the {len(rows)} golden cases")
        if args.sample_seed is None:
            args.sample_seed = random.SystemRandom().randrange(2 ** 32)
        return stratified_sample(rows, args.sample, args.sample_seed)
    return rows[:args.limit] if getattr(args, "limit", None) else rows


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """
    A random draw of `n` cases that still covers every kind of case, returned in
    golden order.

    A plain uniform draw is the wrong tool here. The golden set is 43 answer cases,
    4 clarify and 3 abstain, so a uniform draw of 10 leaves out the clarify and
    abstain cases about half the time - and those are exactly the cases the hard
    gates are made of ("every preserved clarification and abstention edge case
    passes"). A gate with no cases in it passes vacuously, which reads as a success.

    So: take one case from each `category` in turn (largest category first, drawing
    at random inside it) and go round again until `n` are picked. With the canonical
    50 that puts at least one entitlement, law_vs_airline, procedure,
    route_applicability, ambiguous_direction, hybrid_live_data and trap case in any
    sample of 7 or more, and spreads the remainder over the larger categories.

    `seed` is recorded in the result so the draw can be repeated exactly.
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r.get("category") or "uncategorised", []).append(r)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    order = sorted(buckets, key=lambda k: (-len(buckets[k]), k))
    picked: set[str] = set()
    i = 0
    while len(picked) < n and any(buckets[k] for k in order):
        bucket = buckets[order[i % len(order)]]
        if bucket:
            picked.add(bucket.pop()["id"])
        i += 1
    return [r for r in rows if r["id"] in picked]


def estimate_full_run(routing_rows: list[dict], candidates: list[dict], gate: str, n_pilot: int) -> dict:
    """
    The full run's cost, extrapolated from this subset's MEASURED usage: per
    candidate, generator cost per generated case and judge cost per judged case,
    times the number of cases that reach generation in the full routing pass (all
    of them are judged at least once). Noisy on a small pilot; says so.
    """
    reach = [r for r in routing_rows
             if r["generation_attempted"] or (gate == "off" and r["outcome"] == "gated")]
    per = {}
    for c in candidates:
        s = c["summary"]
        gen_case, judge_case = s.get("cost_per_case_usd"), s.get("judge_cost_per_judged_case_usd")
        per[c["name"]] = {
            "generator_cost_per_case_usd": gen_case,
            "judge_cost_per_judged_case_usd": judge_case,
            "generator_usd": None if gen_case is None else round(gen_case * len(reach), 4),
            "judge_usd": None if judge_case is None else round(judge_case * len(reach), 4),
        }
        known = [v for v in (per[c["name"]]["generator_usd"], per[c["name"]]["judge_usd"]) if v is not None]
        per[c["name"]]["total_usd"] = round(sum(known), 4) if len(known) == 2 else None
    totals = [p["total_usd"] for p in per.values()]
    return {
        "basis": (f"measured usage on {n_pilot} pilot case(s) x {len(reach)} cases reaching generation "
                  f"in the full routing pass (gate {gate}); retries and judge re-asks included as "
                  "measured; a 2-case sample is noisy"),
        "full_run_generation_cases": len(reach),
        "per_candidate": per,
        "total_usd": round(sum(totals), 4) if all(t is not None for t in totals) else None,
    }


def run_plan_text(generators: list[dict], judge: tuple[str, str] | None, n_cases: int,
                  judge_req: dict | None = None) -> list[str]:
    lines = [f"cases: {n_cases}"]
    for g in generators:
        where = "no model" if g["provider"] in ("none", "echo") else (
            f"{g['base_url']} ({'local' if is_local_endpoint(g['base_url']) else 'REMOTE, billed/rate-limited'})")
        lines.append(f"generator {g['name']}: {g['model']} @ {where}; "
                     f"up to {n_cases} turns x (1 + retries) calls")
        if g["provider"] not in ("none", "echo"):
            cap = g.get("max_completion_tokens") or "the answer reserve"
            lines.append(f"  request: extra_body={json.dumps(g.get('extra_body') or {})}, "
                         f"{cap_parameter(g['base_url'])}={cap}, served window {g.get('context_window')}, "
                         f"key {g.get('api_key_env') or 'none (local)'}")
            lines.append(f"  client: min {g.get('min_call_interval_s') or 0}s between request starts; "
                         f"Retry-After honoured up to {g.get('retry_max_wait_s', 'LLM_RETRY_MAX_WAIT_S')}s; "
                         "request_too_large never retried")
    if judge:
        per_case = 6
        budget = n_cases * per_case * max(1, len([g for g in generators
                                                  if g["provider"] not in ("none", "echo")]))
        lines.append(f"judge: {judge[1]} @ {judge[0]} "
                     f"({'local' if is_local_endpoint(judge[0]) else 'REMOTE, billed/rate-limited'}); "
                     f"about {per_case} calls per judged case per candidate "
                     f"(~{budget} calls, against a {JUDGE_RPD_LIMIT}/day request cap)")
        pace = judge_min_call_interval_s(judge[0])
        if pace:
            lines.append(f"  client: min {pace}s between request starts "
                         f"(<= {int(60 // pace)} requests/minute), applied to instructor re-asks "
                         "and SDK retries too")
        if judge_req:
            req = {k: v for k, v in judge_req.items() if k != "max_completion_tokens"}
            req[cap_parameter(judge[0])] = judge_req["max_completion_tokens"]
            lines.append(f"  request: {json.dumps(req)}; {JUDGE_STRUCTURED_OUTPUT}; key {JUDGE_ENV_API_KEY}")
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--retrieval-result", default=None,
                    help="retrieval run to take the chunk configuration from "
                         "(default: evals/runs/retrieval/latest.json)")
    ap.add_argument("--allow-smoke-selection", action="store_true",
                    help="accept a non-authoritative retrieval selection (offline smoke runs only)")
    ap.add_argument("--backend", choices=["postgres", "memory"], default="postgres",
                    help="postgres (authoritative, needs --pg-dsn / EVAL_PG_DSN) or memory (smoke)")
    ap.add_argument("--pg-dsn", default=os.environ.get("EVAL_PG_DSN"),
                    help="a DISPOSABLE pgvector database; only the eval_generation_* schema is touched")
    ap.add_argument("--keep-schema", action="store_true")
    ap.add_argument("--generator", action="append", default=None,
                    help="repeatable: none | echo | configured | a name from evals/generator_candidates.json")
    ap.add_argument("--candidates", default=str(CANDIDATES_FILE))
    ap.add_argument("--limit", type=int, default=None, help="first N golden cases only (smoke)")
    ap.add_argument("--pilot", action="store_true",
                    help=f"the fixed pilot cases {', '.join(PILOT_CASE_IDS)} (wiring only)")
    ap.add_argument("--cases", default=None, help="comma-separated golden case ids (wiring only)")
    ap.add_argument("--sample", type=int, default=None,
                    help="N golden cases drawn at random, stratified so every category is "
                         "represented (the clarification and abstention gates need their cases). "
                         "A subset: the result is marked non-authoritative and says so.")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="seed for --sample; omitted means a fresh random seed, which is printed "
                         "and recorded in the result so the same draw can be repeated")
    ap.add_argument("--judge", choices=["api", "fake"], default="api",
                    help="fake is the offline deterministic stand-in, for smoke runs")
    ap.add_argument("--judge-base-url", default=None, help=f"overrides ${JUDGE_ENV_BASE_URL}")
    ap.add_argument("--judge-model", default=None, help=f"overrides ${JUDGE_ENV_MODEL}")
    ap.add_argument("--judge-max-tokens", type=int, default=JUDGE_REQUEST["max_completion_tokens"],
                    help="the judge's completion cap; a non-default value is an override "
                         "(non-authoritative)")
    ap.add_argument("--embedder", choices=["hash", "st"], default=None,
                    help="default: whatever the retrieval run used")
    ap.add_argument("--reranker", choices=["none", "ce"], default=None)
    ap.add_argument("--confirm-paid-calls", action="store_true",
                    help="required when any generator or the judge is a remote API")
    ap.add_argument("--gate", choices=["configured", "off"], default="configured",
                    help="off: candidates generate on every case the gate would refuse (the routing "
                         "pass still records the configured gate); the run is not authoritative")
    args = ap.parse_args(argv)

    from golden.replay import FixtureTool

    from flight_delay.config import get_settings
    from flight_delay.embeddings import build_embedder
    from flight_delay.generation import SYSTEM_PROMPT_SHA256, build_llm
    from flight_delay.pipeline import RagPipeline
    from flight_delay.retrieval import HybridRetriever, build_reranker

    source = (Path(args.retrieval_result) if args.retrieval_result else RETRIEVAL_LATEST).resolve()
    selection = load_retrieval_selection(source, allow_smoke=args.allow_smoke_selection)
    sel = selection["selected"]
    if args.backend == "postgres" and not args.pg_dsn:
        raise SystemExit(
            "the authoritative backend needs a disposable pgvector database: set EVAL_PG_DSN or "
            "pass --pg-dsn (RUNBOOK.md, 'Stage 2'). --backend memory runs an offline smoke test.")

    base = get_settings()
    settings, frozen_differences, cli_overrides = settings_for_selection(base, selection, args)
    generators = resolve_generators(args.generator, load_candidates(Path(args.candidates)), settings)
    routing_only = generators[0]["provider"] == "none"
    all_rows = load_golden_rows()
    rows = select_rows(all_rows, args)
    full_run = len(rows) == len(all_rows)

    judge_target = None if routing_only or args.judge == "fake" else judge_endpoint(args, settings)
    if judge_target:
        if judge_target[1] != JUDGE_EXPECTED_MODEL:
            cli_overrides["judge_model"] = judge_target[1]
        if args.judge_max_tokens != JUDGE_REQUEST["max_completion_tokens"]:
            cli_overrides["judge_max_completion_tokens"] = args.judge_max_tokens
    print("\n".join(run_plan_text(generators, judge_target, len(rows),
                                  judge_request(args) if judge_target else None)))
    if not full_run:
        print(f"  cases: {', '.join(r['id'] for r in rows)}")
        if getattr(args, "sample", None):
            print(f"  drawn by --sample {args.sample} of {len(all_rows)}, stratified by category, "
                  f"seed {args.sample_seed} (pass --sample-seed {args.sample_seed} to repeat it)")
        print("  SUBSET: the run is not authoritative. " + (
            "--pilot selects nothing." if getattr(args, "pilot", False) else
            "A recommendation from it is PROVISIONAL - the hard gates are counted over these "
            "cases only."))
    remote = [g["name"] for g in generators
              if g["provider"] not in ("none", "echo") and not is_local_endpoint(g["base_url"])]
    if judge_target and not is_local_endpoint(judge_target[0]):
        remote.append(f"judge {judge_target[1]}")
    if remote and not args.confirm_paid_calls:
        raise SystemExit(f"remote API calls planned ({', '.join(remote)}). Nothing was sent. "
                         "Re-run with --confirm-paid-calls to proceed.")
    self_judged = [g["name"] for g in generators if judge_target
                   and g.get("model") == judge_target[1]
                   and urlparse(g.get("base_url") or "").hostname == urlparse(judge_target[0]).hostname]
    # Build every generator's settings (and so check its key and window) before any work.
    gen_settings = [generator_settings(settings, g) for g in generators if g["provider"] != "none"]
    if args.gate == "off":
        gen_settings = [gs.model_copy(update={"confidence_gate_enabled": False}) for gs in gen_settings]
        print("  GATE OFF for the candidates (evaluation only; config.py unchanged) -> not authoritative")

    authoritative = run_is_authoritative(
        selection, cli_overrides=cli_overrides, backend=args.backend, limit=not full_run,
        gate=args.gate, judge=args.judge, self_judged=self_judged,
        generators=[] if routing_only else generators)
    for key, (current, frozen) in frozen_differences.items():
        print(f"  frozen from the selection: {key}={frozen!r} (current settings say {current!r})")
    for key, value in cli_overrides.items():
        print(f"  OVERRIDE from the command line: {key}={value!r} -> run is not authoritative")

    judge_llm = judge_model = judge_base_url = judge_meter = None
    if not routing_only:
        judge_llm, judge_model, judge_base_url, judge_meter = build_judge(args, settings)
        if args.judge != "fake":
            print(f"judge preflight: one structured call to {judge_model}...", flush=True)
            preflight_judge(judge_llm, judge_model, judge_base_url)
            print("  judge preflight OK (it answers and returns a valid structured object)")
    judge_prices = JUDGE_PRICES if judge_model == JUDGE_EXPECTED_MODEL else None

    print(f"selected chunk configuration: {sel['chunk_target_tokens']} tokens, "
          f"{sel['chunk_overlap_pct']}% overlap   (from {display_path(source)})")
    embedder = build_embedder(settings)
    schema = f"{SCHEMA_PREFIX}_{sel['chunk_target_tokens']}_{sel['chunk_overlap_pct']}"
    t_start = time.time()
    if args.backend == "postgres":
        print(f"indexing into schema {schema} ({settings.embedder} embedder)...", flush=True)
        store, chunks = build_pg_index(settings, embedder, args.pg_dsn, schema)
    else:
        store, chunks = build_index(settings, embedder)
    retriever = CachingRetriever(HybridRetriever(store, embedder, build_reranker(settings), settings))
    index_s = time.time() - t_start

    try:
        # Pass 0: routing, retrieval and the gate, no model. FixtureTool(None) is a
        # placeholder; run_conversation installs each case's own. The real AirLabs
        # tool is never constructed on this path.
        # The routing pass always covers every case (it is free), so a pilot can
        # extrapolate the full run's cost from its measured per-case usage.
        print(f"index: {len(chunks)} chunks ({index_s:.0f}s). Routing pass over {len(all_rows)} cases...",
              flush=True)
        routing_pipe = RagPipeline(retriever, NoGeneration(), FixtureTool(None), settings, store=None)
        routing_rows = run_cases(routing_pipe, all_rows)
        calibration = gate_calibration(routing_rows, settings)

        results = []
        for gen, gs in zip([g for g in generators if g["provider"] != "none"], gen_settings, strict=True):
            print(f"generating with {gen['name']}...", flush=True)
            pipe = RagPipeline(retriever, build_llm(gs), FixtureTool(None), gs, store=None)
            t0 = time.time()
            case_rows = run_cases(pipe, rows, served_window=gen.get("context_window"))
            gen_s = time.time() - t0
            print(f"  {gen['name']}: {deterministic_metrics(case_rows)['outcomes']} ({gen_s:.0f}s); "
                  f"judging with {judge_model}...", flush=True)
            asyncio.run(_score(case_rows, judge_plan(case_rows), judge_llm,
                               progress=lambda i, n: print(f"    judged {i}/{n}", flush=True)
                               if i % 10 == 0 or i == n else None, meter=judge_meter))
            summary = {**deterministic_metrics(case_rows), **ragas_summary(case_rows),
                       **usage_summary(case_rows, gen), **judge_usage_summary(case_rows, judge_prices)}
            public = {k: v for k, v in gen.items() if k != "api_key"}
            public["api_key_source"] = (candidate_api_key(gen)[1]
                                        if gen["provider"] != "echo" else None)
            for r in case_rows:
                r["evidence_chars"] = sum(len(e) for e in r.pop("evidence"))
            results.append({"name": gen["name"], "generator": public, "generation_s": round(gen_s, 1),
                            "summary": summary, "items": case_rows})
    finally:
        if args.backend == "postgres" and not args.keep_schema:
            drop_schema(args.pg_dsn, schema)

    for r in routing_rows:
        r.pop("evidence", None)
    run = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "complete": True,
        "stage": "generation",
        "duration_s": round(time.time() - t_start, 1),
        "authoritative": authoritative,
        "selected": sel,
        "retrieval_result": display_path(source),
        "retrieval_config_digest": selection.get("config_digest"),
        "frozen_from_selection": {k: v[1] for k, v in frozen_differences.items()},
        "cli_overrides": cli_overrides,
        "config": {
            "backend": args.backend,
            **(pg_versions(args.pg_dsn) if args.backend == "postgres" else {}),
            "embedder": settings.embedder, "embedding_model": settings.embedding_model,
            "reranker": settings.reranker, "reranker_model": settings.reranker_model,
            "final_k": settings.final_k, "candidates_k": settings.candidates_k,
            "balance_sources": settings.balance_sources,
            "context_token_budget": settings.context_token_budget,
            # Common to every candidate: the input-assembly budget, not a served window.
            "input_assembly_budget_tokens": settings.llm_context_window,
            "context_answer_reserve_tokens": settings.context_answer_reserve_tokens,
            "max_assembled_prompt_tokens": max_prompt_tokens(settings),
            "llm_temperature": settings.llm_temperature,
            "max_validation_retries": settings.max_validation_retries,
            "confidence_gate_enabled": settings.confidence_gate_enabled,
            "gate_for_candidates": args.gate,
            "n_chunks": len(chunks), "index_s": round(index_s, 1),
            "distinct_retrievals": retriever.misses,
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        },
        "judge": None if routing_only else {
            # The URL, never the key. Result files are meant to be shareable.
            "model": judge_model, "base_url": judge_base_url,
            "expected_model": JUDGE_EXPECTED_MODEL,
            "request": None if args.judge == "fake" else {
                **{k: v for k, v in judge_request(args).items() if k != "max_completion_tokens"},
                cap_parameter(judge_base_url): args.judge_max_tokens},
            "structured_output": None if args.judge == "fake" else JUDGE_STRUCTURED_OUTPUT,
            "api_key_source": None if args.judge == "fake" else JUDGE_ENV_API_KEY,
            "prices": judge_prices,
            "usage_total": summarize_calls(judge_meter.calls, judge_prices) if judge_meter else None,
            "is_offline_stand_in": args.judge == "fake",
            "self_evaluation": self_judged,
        },
        "evidence_scope": EVIDENCE_SCOPE,
        "golden_version": golden_version(),
        "golden_digest": golden_digest(),
        "corpus_digest": corpus_digest(),
        "prompt_digest": prompt_digest(),
        "n_cases": len(rows),
        "case_ids": [r["id"] for r in rows],
        "full_run": full_run,
        "sample": ({"n": args.sample, "seed": args.sample_seed, "strategy": "stratified by category",
                    "of": len(all_rows)} if getattr(args, "sample", None) else None),
        "routing_cases": len(routing_rows),
        "reference_status": "draft",
        "gate_calibration": calibration,
        "routing_pass": routing_rows,
        "candidates": results,
        "full_run_estimate": (None if full_run or not results
                              else estimate_full_run(routing_rows, results, args.gate, len(rows))),
        "recommendation": (recommend_generator(
            results, settings, full_run=full_run,
            wiring_only=bool(getattr(args, "pilot", False)),
            basis=selection_basis(rows, all_rows, args)) if results else None),
    }

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    prefix = "calibration-" if routing_only else ""
    path = OUT_DIR / f"{prefix}{stamp}.json"
    payload = json.dumps(run, indent=2, default=str)
    atomic_write_text(path, payload)
    handoff = None
    if not routing_only:
        handoff = LATEST if authoritative else LATEST_SMOKE
        atomic_write_text(handoff, payload)

    print_report(run)
    print(f"\nsaved: {display_path(path)}")
    if handoff:
        print(f"saved: {display_path(handoff)}"
              + ("" if authoritative else "   (NOT authoritative: smoke, override, --limit or self-judged)"))
    print("Nothing in config.py or .env was changed. Adopting a generator means setting ALL of its "
          "printed production settings, including LLM_MAX_COMPLETION_TOKENS (its tested cap).")


if __name__ == "__main__":
    main()
