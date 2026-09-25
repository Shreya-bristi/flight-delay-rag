#!/usr/bin/env python3
"""
choose the chunk configuration, using deterministic retrieval metrics.

METRICS : section_recall@3, MRR,nDCG@5
  
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "evals"):
    sys.path.insert(0, str(p))

OUT_DIR = ROOT / "evals" / "runs" / "retrieval"
LATEST = OUT_DIR / "latest.json"
# Where a completed run that is NOT a measurement (memory backend, stand-in
# embedder or reranker, or --limit) is handed off. Stage 2 reads it only when told to.
LATEST_SMOKE = OUT_DIR / "latest-smoke.json"
GOLDEN = ROOT / "evals" / "golden_set.jsonl"

SHINGLE = 5
SECTION_RECALL_K = 3
NDCG_K = 5
RECALL_TOLERANCE = 0.02
SCHEMA_PREFIX = "eval_retrieval"

SELECTION_RULE = (
    "evidence_recall within 0.02 of best -> highest MRR -> lowest context_tokens "
    "-> smaller chunk."
)
LEXICAL_POSTGRES = ("PostgreSQL full-text search: OR of plainto_tsquery('english') lexemes, "
                    "ranked by ts_rank_cd normalisation 1 (not BM25)")
LEXICAL_MEMORY = "MemoryStore Okapi BM25 (test double, not the production lexical ranker)"


# --------------------------------------------------------------------------
# Size-independent text metrics
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9§$€£]+(?:[.,][0-9]+)*")


def shingles(text: str, n: int = SHINGLE) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def evidence_metrics(gold_texts: list[str], context_texts: list[str]) -> dict[str, float]:
    """
    Coverage of the labelled gold passages by the kept context, by single overlap.

    Size-independent on purpose: the golden set's gold passages are resolved
    against parsed sections, and the TEXT of a clause is the same however it is split.
    """
    ctx = set().union(*(shingles(t) for t in context_texts)) if context_texts else set()
    gold_sets = [s for s in (shingles(t) for t in gold_texts) if s]
    if not gold_sets:
        return {"evidence_recall": 0.0, "evidence_hit_rate": 0.0, "context_precision": 0.0}
    coverage = [len(g & ctx) / len(g) for g in gold_sets]
    all_gold = set().union(*gold_sets)
    return {
        "evidence_recall": statistics.fmean(coverage),
        "evidence_hit_rate": sum(c >= 0.5 for c in coverage) / len(coverage),
        "context_precision": (len(ctx & all_gold) / len(ctx)) if ctx else 0.0,
    }


def dedupe_preserving_order(items: list[str]) -> list[str]:
    """Collapse repeated doc#section values, keeping the first occurrence (for display)."""
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def section_recall_at_k(sections: list[str], gold: set[str], k: int = SECTION_RECALL_K) -> float:
    """Fraction of gold doc#section values present in the first k retrieved CHUNKS."""
    if not gold:
        return 0.0
    return len(gold & set(sections[:k])) / len(gold)


def required_evidence(premises: list[dict], kept) -> dict:
    """
    Two of the three Session 24 numbers, per case (None for a case with no premises):

      required_complete           1.0 only if EVERY premise (golden/required.py) was
                                  proved by at least one of its accepted sources.
                                  evidence_recall can read 0.5 while the one clause the
                                  answer turns on is missing (live-01: Art 6 kept, Art 7 not).
      primary_authority_kept      share of the premises that have a binding regulation
                                  among their sources where that regulation was kept,
                                  rather than only a regulator's guidance repeating it.

    A premise is proved by ANY of its sources, because the same rule is often stated
    twice in the corpus (DOT Q&A / Part 260, CAA / UK261, guidelines / EU261). The
    third number, evidence_recall, is unchanged and measured elsewhere.
    """
    if not premises:
        return {"required_complete": None, "required_missing": [], "primary_authority_kept": None}
    have = {f"{c.doc_id}#{c.section_id}" for c in kept}
    missing = [g["any_of"] for g in premises if not (set(g["any_of"]) & have)]
    with_primary = [g for g in premises if g["primary"]]
    primary_kept = ([1.0 for g in with_primary if set(g["primary"]) & have]
                    if with_primary else [])
    return {
        "required_complete": 0.0 if missing else 1.0,
        "required_missing": [g[0] for g in missing],
        "primary_authority_kept": (round(len(primary_kept) / len(with_primary), 4)
                                   if with_primary else None),
    }


def mrr(sections: list[str], gold: set[str]) -> float:
    """Reciprocal position of the first retrieved chunk whose doc#section is gold."""
    if not gold:
        return 0.0
    return next((1.0 / i for i, s in enumerate(sections, start=1) if s in gold), 0.0)


def _dcg(relevances: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def ndcg_at_k(sections: list[str], gold: set[str], k: int = NDCG_K) -> float:
    """
    Binary-relevance nDCG over the first k CHUNK positions.

    The cutoff is applied to chunk positions FIRST. A chunk gains 1 if its
    doc#section is gold and no higher-ranked chunk has already been credited for
    that section; repeats gain 0. So a gold section repeated at positions 1-3
    cannot push DCG above the ideal (the old nDCG > 1.0 bug), and a gold chunk at
    position 6 behind five copies of one wrong section scores 0 (the old
    dedupe-before-cutoff version gave it 0.63).
    """
    if not gold:
        return 0.0
    credited: set[str] = set()
    rels = []
    for s in sections[:k]:
        hit = s in gold and s not in credited
        rels.append(1 if hit else 0)
        if hit:
            credited.add(s)
    idcg = _dcg([1] * min(len(gold), k))
    return min(1.0, _dcg(rels) / idcg) if idcg > 0 else 0.0


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_digest(path: Path) -> str:
    """SHA-256 of the bytes on disk (first 16 hex chars), or 'missing'."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def golden_digest() -> str:
    return file_digest(GOLDEN)


def golden_version() -> int:
    from golden.spec import GOLDEN_VERSION

    return GOLDEN_VERSION


def prompt_digest() -> str:
    """The exact system prompt bytes: a changed prompt is a different system."""
    from flight_delay.generation import SYSTEM_PROMPT_PATH

    return file_digest(SYSTEM_PROMPT_PATH)


def corpus_digest() -> str:
    """Manifest rows plus source CONTENT hashes; one definition, shared with the indexer."""
    import index_corpus

    return index_corpus.corpus_digest()


def display_path(path: Path) -> str:
    """Repo-relative when possible; the absolute path otherwise. Never raises."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def atomic_write_text(path: Path, text: str) -> None:
    """Write to a sibling temp file, then replace: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_golden_rows() -> list[dict]:
    """The canonical golden set, refused unless it is exactly the builder's canonical cases."""
    from golden.spec import CANONICAL_CASES

    rows = [json.loads(line) for line in
            GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if ids != list(CANONICAL_CASES):
        raise SystemExit(
            f"{GOLDEN.relative_to(ROOT)} is not the canonical {len(CANONICAL_CASES)}-case set; "
            "rebuild it with evals/build_golden_set.py")
    return rows


def split_items(rows: list[dict], limit: int | None = None) -> tuple[list[dict], list[dict]]:
    """
    (scored, excluded). Scored = answerable cases carrying labelled gold context.

    Trap/decline cases and clarify cases have no gold evidence to retrieve — a
    correct decline or a correct clarifying question retrieves nothing — so they
    would drag every average toward zero for the wrong reason. They are excluded
    from the retrieval means and counted in the report instead.
    """
    scored, excluded = [], []
    for r in rows:
        action = (r.get("expected_behavior") or {}).get("action", "answer")
        if not r.get("answerable"):
            excluded.append({"id": r["id"], "reason": f"not answerable ({action})"})
        elif not r.get("context"):
            excluded.append({"id": r["id"], "reason": "no labelled gold context"})
        else:
            scored.append(r)
    if limit:
        dropped = scored[limit:]
        scored = scored[:limit]
        excluded += [{"id": r["id"], "reason": "beyond --limit"} for r in dropped]
    return scored, excluded


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------


def _chunks_for(settings):
    import index_corpus

    with contextlib.redirect_stdout(io.StringIO()):
        chunks, problems = index_corpus.build_chunks(settings)
    if problems:
        raise SystemExit(f"corpus did not build cleanly: {problems}")
    return chunks


def build_index(settings, embedder):
    """One in-memory index at this settings' chunk size (smoke path; Stage 2 uses it too)."""
    from flight_delay.store import MemoryStore

    chunks = _chunks_for(settings)
    store = MemoryStore()
    store.upsert(chunks, embedder.embed_documents([c.embed_text for c in chunks]))
    return store, chunks


def schema_name(size: int, overlap: int) -> str:
    return f"{SCHEMA_PREFIX}_{size}_{overlap}"


# Schemas the two evaluation stages create, and the only ones either may drop.
EVAL_SCHEMA_PREFIXES = (SCHEMA_PREFIX + "_", "eval_generation_")


def drop_schema(dsn: str, schema: str) -> None:
    import psycopg

    if not schema.startswith(EVAL_SCHEMA_PREFIXES):
        raise ValueError(f"refusing to drop a schema this evaluation did not create: {schema}")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def build_pg_index(settings, embedder, dsn: str, schema: str):
    """
    Index the corpus into `schema` the way scripts/index_corpus.py does: staging
    table, validation, activation. Returns (store, chunks).
    """
    import index_corpus

    from flight_delay.store import PgVectorStore

    chunks = _chunks_for(settings)
    drop_schema(dsn, schema)
    store = PgVectorStore(dsn, schema=schema, hnsw_ef_search=settings.hnsw_ef_search)
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    staging = store.begin_staging(embedder.dim)
    staging.add(chunks, vectors)
    staging.activate(index_corpus.index_identity(settings, embedder, chunks),
                     expected_chunks=len(chunks),
                     required_docs=[e["doc_id"] for e in index_corpus.CORPUS])
    with store._conn() as c:
        c.execute("ANALYZE chunks")
    return store, chunks


def pg_versions(dsn: str) -> dict:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as c:
        server = c.execute("SHOW server_version").fetchone()[0]
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        vector = c.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
    return {"postgres": server, "pgvector": vector}


# --------------------------------------------------------------------------
# Production pipeline, stopped at the built context
# --------------------------------------------------------------------------


def _stage_one_pipeline(store, embedder, reranker, settings):
    """
    RagPipeline wired like build_pipeline(), except: the generator stops the turn
    once build_context() has run, and the confidence gate is off (its decision is
    recomputed on the same chunks and reported instead).
    """
    from flight_delay.generation import LLMError
    from flight_delay.pipeline import RagPipeline
    from flight_delay.retrieval import HybridRetriever

    class StageOneStop(LLMError):
        kind = "stage1_no_generation"

    class NoGeneration:
        """No model call in Stage 1: the turn ends with the context built."""

        def complete(self, system, user):  # noqa: ARG002
            raise StageOneStop("stage 1 measures retrieval and context only")

    class RecordingRetriever(HybridRetriever):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.calls: list[tuple[str, dict | None, object]] = []

        def retrieve(self, query, **kw):
            res = super().retrieve(query, **kw)
            self.calls.append((query, kw.get("flt"), res))
            return res

    no_gate = settings.model_copy(update={"confidence_gate_enabled": False})
    retriever = RecordingRetriever(store, embedder, reranker, no_gate)
    return RagPipeline(retriever, NoGeneration(), None, no_gate, store=None), retriever


def model_windows(embedder, reranker) -> dict:
    """The real models' tokenizers and input windows, when real models are in use."""
    out = {}
    model = getattr(embedder, "model", None)
    if model is not None:
        out["embed"] = (model.tokenizer, int(model.max_seq_length))
    ce = getattr(reranker, "model", None)
    if ce is not None:
        out["rerank"] = (ce.tokenizer, int(getattr(ce, "max_seq_length", None) or ce.max_length))
    return out


def _share_over(tokenizer, window: int, inputs: list) -> float | None:
    if not inputs:
        return None
    n = 0
    for x in inputs:
        enc = tokenizer(*x, verbose=False) if isinstance(x, tuple) else tokenizer(x, verbose=False)
        n += len(enc["input_ids"]) > window
    return round(n / len(inputs), 4)


def _pct(values: list[float], q: float):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def chunk_stats(chunks, settings, windows: dict) -> dict:
    toks = [c.token_estimate for c in chunks]
    embed = windows.get("embed")
    return {
        "n_chunks": len(chunks),
        "tokens_median": _pct(toks, 0.5),
        "tokens_p90": _pct(toks, 0.9),
        "tokens_max": max(toks),
        "chunks_over_target": sum(t > settings.chunk_target_tokens for t in toks),
        # Header + body + special tokens, with the embedding model's tokenizer.
        "embed_truncated_pct": (_share_over(embed[0], embed[1], [c.embed_text for c in chunks])
                                if embed else None),
    }


def gold_passages_in_one_chunk(items, chunks) -> float | None:
    """Share of gold passages whose whole text sits inside a single chunk at this size."""
    by_doc: dict[str, list[str]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(" ".join(c.text.split()))
    total = inside = 0
    for it in items:
        for g in it["context"]:
            text = " ".join(g["text"].split())
            total += 1
            inside += any(text in t for t in by_doc.get(g["doc_id"], ()))
    return round(inside / total, 4) if total else None


# --------------------------------------------------------------------------
# Dense search: production vs exact vs forced HNSW (Postgres only)
# --------------------------------------------------------------------------


def ann_diagnostics(store, dsn: str, queries: list[tuple[list, dict | None]], k: int) -> dict:
    """
    For each (query vector, filter) the pipeline searched with: production
    dense_search, the same query as an exact scan, and dense_search forced through
    the HNSW index (sequential and bitmap scans disabled). Fresh connections per
    query, so no prepared plan made under one setting is reused under another.
    """
    import psycopg

    from flight_delay.store import PgVectorStore, _filter_sql, _vec_literal

    def connect(extra: str = ""):
        return psycopg.connect(dsn, autocommit=True, prepare_threshold=None,
                               options=f"-c search_path={store.schema},public {extra}".strip())

    forced = PgVectorStore(dsn, schema=store.schema, hnsw_ef_search=store.hnsw_ef_search)
    forced._conn = lambda: connect("-c enable_seqscan=off -c enable_bitmapscan=off")

    rows = {"production": [], "forced_hnsw": [], "forced_hnsw_pgvector_defaults": []}
    plans: Counter = Counter()
    for qvec, flt in queries:
        params: list = []
        where = _filter_sql(flt, params)
        lit = _vec_literal(qvec)
        sql = (f"SELECT chunk_id FROM chunks {where} "
               "ORDER BY embedding <=> %s::vector LIMIT %s")
        with connect("-c enable_indexscan=off") as c:
            exact = {r[0] for r in c.execute(sql, [*params, lit, k]).fetchall()}
        for name, st in (("production", store), ("forced_hnsw", forced), ("forced_hnsw_pgvector_defaults", None)):
            t0 = time.perf_counter()
            if st is None:   # what the index returns without dense_search's search settings
                with connect("-c enable_seqscan=off -c enable_bitmapscan=off") as c:
                    got = [r[0] for r in c.execute(sql, [*params, lit, k]).fetchall()]
            else:
                got = [cid for cid, _ in st.dense_search(qvec, k, flt)]
            ms = (time.perf_counter() - t0) * 1000
            rows[name].append({"n": len(got), "expected": len(exact), "ms": ms,
                               "recall": len(set(got) & exact) / len(exact) if exact else 1.0})
        with connect() as c, c.transaction():
            c.execute("SELECT set_config('hnsw.ef_search', %s, true), "
                      "set_config('hnsw.iterative_scan', 'strict_order', true)",
                      [str(max(store.hnsw_ef_search, k))])
            plan = [r[0] for r in c.execute("EXPLAIN " + sql, [*params, lit, k]).fetchall()]
        scans = [re.sub(r"\s+\(cost.*", "", ln).strip(" ->") for ln in plan if "Scan" in ln]
        plans[scans[0] if scans else "no scan"] += 1

    def summary(r):
        return {"recall_vs_exact_mean": round(statistics.fmean(x["recall"] for x in r), 4),
                "recall_vs_exact_min": round(min(x["recall"] for x in r), 4),
                "short_lists": sum(x["n"] < x["expected"] for x in r),
                "median_ms": round(statistics.median(x["ms"] for x in r), 2)} if r else {}

    return {"k": k, "queries": len(queries), "production": summary(rows["production"]),
            "forced_hnsw": summary(rows["forced_hnsw"]),
            "forced_hnsw_pgvector_defaults": summary(rows["forced_hnsw_pgvector_defaults"]),
            "production_plans": dict(plans)}


# --------------------------------------------------------------------------
# One chunk configuration
# --------------------------------------------------------------------------

PRIMARY_COLUMNS = [
    ("chunk_tokens", "chunk_target_tokens", "{}"),
    ("overlap_pct", "chunk_overlap_pct", "{}"),
    ("evidence_recall", "evidence_recall", "{:.3f}"),
    ("evidence_hit_rate", "evidence_hit_rate", "{:.3f}"),
    ("required_complete", "required_premise_complete", "{:.3f}"),
    ("primary_authority", "primary_authority_coverage", "{:.3f}"),
    ("section_recall@3", "section_recall@3", "{:.3f}"),
    ("MRR", "mrr", "{:.3f}"),
    ("nDCG@5", "ndcg@5", "{:.3f}"),
    ("context_precision", "context_precision", "{:.3f}"),
    ("context_tokens", "context_tokens", "{:.0f}"),
    ("retrieval_ms", "retrieval_ms", "{:.0f}"),
]

DIAGNOSTIC_COLUMNS = [
    ("chunk_tokens", "chunk_target_tokens", "{}"),
    ("overlap", "chunk_overlap_pct", "{}"),
    ("chunks", "n_chunks", "{}"),
    ("tok med/p90/max", "tokens_dist", "{}"),
    ("embed>win", "embed_truncated_pct", "{:.1%}"),
    ("pair>win", "reranker_pair_truncated_pct", "{:.1%}"),
    ("gold in 1 chunk", "gold_passages_in_one_chunk", "{:.1%}"),
    ("kept/retrieved", "kept_of_retrieved", "{:.2f}"),
    ("governing kept", "governing_kept_share", "{:.1%}"),
    ("off-carrier", "off_carrier_chunks", "{}"),
    ("dup-section", "duplicate_section_chunks", "{}"),
    ("gate abstain", "gate_would_abstain", "{}"),
]


def evaluate_config(base, size: int, overlap: int, items, embedder, reranker, *,
                    backend: str = "postgres", dsn: str | None = None, keep_schema: bool = False):
    """Index at (size, overlap), run every case through the production pipeline, score."""
    from golden.replay import run_conversation

    from flight_delay.confidence import ConfidenceGate
    from flight_delay.config import revalidate

    settings = revalidate(
        base.model_copy(update={"chunk_target_tokens": size, "chunk_overlap_pct": overlap}))
    t0 = time.time()
    schema = schema_name(size, overlap)
    if backend == "postgres":
        store, chunks = build_pg_index(settings, embedder, dsn, schema)
    else:
        store, chunks = build_index(settings, embedder)
    index_s = time.time() - t0
    pipe, retriever = _stage_one_pipeline(store, embedder, reranker, settings)
    gate = ConfidenceGate(min_rerank_score=settings.min_rerank_score,
                          min_score_margin=settings.min_score_margin,
                          min_grounding_ratio=settings.min_grounding_ratio, enabled=True)
    windows = model_windows(embedder, reranker)

    rows, pairs, ann_queries, top_scores, all_scores = [], [], [], [], []
    for item in items:
        retriever.calls.clear()
        answer = run_conversation(pipe, item)
        built = answer.built_context
        kept = list(built.used_chunks) if built is not None else []
        call = retriever.calls[-1] if retriever.calls else None
        query, flt, res = call if call else (item["question"], None, None)
        ranked = list(res.ranked) if res is not None else []
        sections = [f"{c.doc_id}#{c.section_id}" for c in ranked]
        gold = set(item["gold_doc_sections"])

        governing = [j for j in ((flt or {}).get("jurisdiction_governing") or ())]
        kept_gov = {c.jurisdiction for c in kept if c.source_class == "government"}
        carrier = (item.get("scenario") or {}).get("airline") or ""
        # The gate sees the passenger's (effective) question, as in production, not the
        # structured search query (Session 22).
        decision = (gate.evaluate(answer.question or query, list(res.chunks))
                    if res is not None else None)
        if ranked:
            pairs += [(query, c.embed_text) for c in ranked]
            scores = [c.rerank_score for c in ranked if c.rerank_score is not None]
            all_scores += scores
            if scores:
                top_scores.append(scores[0])
        if res is not None and backend == "postgres":
            ann_queries.append((embedder.embed_query(query), flt))

        rows.append({
            "id": item["id"],
            "category": item["category"],
            "turn_outcome": answer.outcome,
            "context_built": built is not None,
            **evidence_metrics([c["text"] for c in item["context"]], [c.text for c in kept]),
            "section_recall@3": section_recall_at_k(sections, gold),
            "mrr": mrr(sections, gold),
            "ndcg@5": ndcg_at_k(sections, gold),
            "context_tokens": built.estimated_tokens if built is not None else 0,
            "retrieval_ms": round(sum((res.timings_ms if res else {}).values()), 1),
            "n_ranked": len(ranked),
            "n_retrieved": len(res.chunks) if res is not None else 0,
            "n_kept": len(kept),
            "governing": governing,
            "governing_kept": (all(j in kept_gov for j in governing) if governing else None),
            "off_carrier_chunks": sum(1 for c in kept if c.airline_iata and c.airline_iata != carrier),
            "duplicate_section_chunks": len(kept) - len({(c.doc_id, c.section_id) for c in kept}),
            "gate_would_abstain": (not decision.should_answer) if decision else None,
            "gate_reasons": decision.reasons if decision and not decision.should_answer else [],
            "top_rerank_score": ranked[0].rerank_score if ranked else None,
            "retrieved_sections": dedupe_preserving_order(sections)[:10],
            "kept_sections": dedupe_preserving_order([f"{c.doc_id}#{c.section_id}" for c in kept]),
            **required_evidence(item.get("required_premises") or [], kept),
        })

    summary = {
        "chunk_target_tokens": size,
        "chunk_overlap_pct": overlap,
        "index_s": round(index_s, 1),
        **chunk_stats(chunks, settings, windows),
    }
    summary["tokens_dist"] = f"{summary['tokens_median']}/{summary['tokens_p90']}/{summary['tokens_max']}"
    rerank = windows.get("rerank")
    summary["reranker_pair_truncated_pct"] = (_share_over(rerank[0], rerank[1], pairs)
                                              if rerank else None)
    summary["gold_passages_in_one_chunk"] = gold_passages_in_one_chunk(items, chunks)
    for key in ("evidence_recall", "evidence_hit_rate", "section_recall@3", "mrr", "ndcg@5",
                "context_precision", "context_tokens", "retrieval_ms"):
        summary[key] = _mean(rows, key)
    summary["required_premise_complete"] = _mean(rows, "required_complete")
    summary["primary_authority_coverage"] = _mean(rows, "primary_authority_kept")
    summary["required_incomplete_cases"] = [r["id"] for r in rows if r.get("required_complete") == 0.0]
    retrieved = sum(r["n_retrieved"] for r in rows)
    summary["kept_of_retrieved"] = round(sum(r["n_kept"] for r in rows) / retrieved, 4) if retrieved else None
    with_gov = [r for r in rows if r["governing_kept"] is not None]
    summary["governing_kept_share"] = (round(sum(r["governing_kept"] for r in with_gov) / len(with_gov), 4)
                                       if with_gov else None)
    summary["governing_missing_cases"] = [r["id"] for r in with_gov if not r["governing_kept"]]
    summary["off_carrier_chunks"] = sum(r["off_carrier_chunks"] for r in rows)
    summary["duplicate_section_chunks"] = sum(r["duplicate_section_chunks"] for r in rows)
    summary["gate_would_abstain"] = sum(bool(r["gate_would_abstain"]) for r in rows)
    summary["reranker_scores"] = {
        "activation": getattr(reranker, "score_activation", "none (no reranker)"),
        "top1_p10_p50_p90": [_round(_pct(top_scores, q)) for q in (0.1, 0.5, 0.9)],
        "all_min_max": [_round(min(all_scores)), _round(max(all_scores))] if all_scores else None,
    }
    # A good mean can hide a total miss on one legal exception; name them.
    summary["zero_evidence_cases"] = [r["id"] for r in rows if r["evidence_recall"] == 0.0]
    summary["no_context_cases"] = [f"{r['id']} ({r['turn_outcome']})" for r in rows if not r["context_built"]]
    if backend == "postgres":
        summary["dense_search"] = ann_diagnostics(store, dsn, ann_queries, settings.candidates_k)
        if not keep_schema:
            drop_schema(dsn, schema)
    return summary, rows


def _round(v):
    return None if v is None else round(v, 3)


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.fmean(vals), 4) if vals else None


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def select_configuration(summaries: list[dict]) -> dict:
    """
    One transparent rule, deliberately not a weighted composite score:

      1. highest mean evidence_recall
      2. keep every configuration within 0.02 of it
      3. of those, highest MRR
      4. still tied -> lowest mean context_tokens
      5. still tied -> smaller chunk size

    A weighted score would hide the trade-off being made inside a constant
    nobody can argue with. This rule can be disagreed with line by line.
    """
    if not summaries:
        raise ValueError("no configurations to select from")
    best_recall = max(s["evidence_recall"] for s in summaries)
    tied = [s for s in summaries if s["evidence_recall"] >= best_recall - RECALL_TOLERANCE]
    return min(tied, key=lambda s: (-s["mrr"], s["context_tokens"], s["chunk_target_tokens"]))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def format_table(summaries: list[dict], columns=PRIMARY_COLUMNS) -> str:
    header = [c[0] for c in columns]
    body = []
    for s in summaries:
        body.append([("n/a" if s.get(key) is None else fmt.format(s[key]))
                     for _, key, fmt in columns])
    widths = [max(len(header[i]), *(len(r[i]) for r in body)) if body else len(header[i])
              for i in range(len(header))]
    lines = ["  ".join(h.rjust(w) for h, w in zip(header, widths, strict=True)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)) for row in body]
    return "\n".join(lines)


def print_report(run: dict) -> None:
    cfg = run["config"]
    print("\n" + "=" * 78)
    print("RETRIEVAL EVALUATION")
    print("=" * 78)
    embedder = cfg["embedding_model"] if cfg["embedder"] == "st" else "hash (smoke)"
    reranker = cfg["reranker_model"] if cfg["reranker"] == "ce" else "none"
    print(f"cases {run['n_cases']} scored, {len(run['excluded'])} excluded  |  backend {cfg['backend']}"
          f"  |  embedder {embedder}  |  reranker {reranker}  |  k {cfg['candidates_k']}->{cfg['final_k']}"
          f"  |  golden v{run['golden_version']}")
    print(f"lexical: {cfg['lexical']}")
    print(f"chunk sizes counted in: {cfg['chunk_token_counter']}")
    print()
    print(format_table(run["configurations"]))
    print()
    print("Diagnostics (not used for selection)")
    print(format_table(run["configurations"], DIAGNOSTIC_COLUMNS))
    for s in run["configurations"]:
        tag = f"{s['chunk_target_tokens']}tok/{s['chunk_overlap_pct']}%"
        dense = s.get("dense_search")
        if dense:
            print(f"  {tag} dense search vs exact ({dense['queries']} queries, k={dense['k']}): "
                  f"production {dense['production']}; forced HNSW {dense['forced_hnsw']}; "
                  f"forced HNSW with pgvector defaults {dense['forced_hnsw_pgvector_defaults']}; "
                  f"production plans {dense['production_plans']}")
        print(f"  {tag} reranker scores ({s['reranker_scores']['activation']}): "
              f"top-1 p10/p50/p90 {s['reranker_scores']['top1_p10_p50_p90']}, "
              f"all min/max {s['reranker_scores']['all_min_max']}")
    print()
    sel = run["selected"]
    print(f"SELECTED CHUNK CONFIGURATION: {sel['chunk_target_tokens']} tokens, "
          f"{sel['chunk_overlap_pct']}% overlap")
    print(f"Selection: {run['selection_rule']}")
    for s in run["configurations"]:
        tag = f"{s['chunk_target_tokens']}tok/{s['chunk_overlap_pct']}%"
        if s["zero_evidence_cases"]:
            print(f"  WARNING {tag}: evidence_recall 0 on {', '.join(s['zero_evidence_cases'])}")
        if s.get("required_incomplete_cases"):
            print(f"  {tag}: required evidence incomplete on "
                  f"{len(s['required_incomplete_cases'])}: {', '.join(s['required_incomplete_cases'])}")
        if s["no_context_cases"]:
            print(f"  WARNING {tag}: no context built for {', '.join(s['no_context_cases'])} "
                  "(scored as 0, not dropped)")
        if s.get("governing_missing_cases"):
            print(f"  {tag}: a governing regime missing from the kept context: "
                  f"{', '.join(s['governing_missing_cases'])}")
    if run["excluded"]:
        print(f"  excluded from the averages ({len(run['excluded'])}): "
              f"{', '.join(e['id'] for e in run['excluded'])}")
    if not run["authoritative"]:
        print("  NOT AUTHORITATIVE: memory backend, a stand-in model or --limit. "
              "These numbers are not reportable measurements.")


# --------------------------------------------------------------------------
# Result file
# --------------------------------------------------------------------------

REQUIRED_KEYS = (
    "timestamp", "complete", "authoritative", "config", "config_digest", "configurations",
    "items", "selected", "selection_rule", "canonical_case_ids", "golden_version",
    "golden_digest", "corpus_digest", "prompt_digest", "n_cases", "excluded",
)
BOUNDED_METRICS = ("evidence_recall", "evidence_hit_rate", "section_recall@3",
                   "mrr", "ndcg@5", "context_precision")
# Measured only with real model tokenizers; None on a smoke run.
OPTIONAL_BOUNDED_METRICS = ("embed_truncated_pct", "reranker_pair_truncated_pct",
                            "gold_passages_in_one_chunk", "governing_kept_share",
                            "required_premise_complete", "primary_authority_coverage")


def schema_errors(run: dict) -> list[str]:
    """
    Everything a completed selection must carry, checked BEFORE it is written.

    An interrupted or malformed run must never be mistaken for a finished
    selection: generation evaluation reads latest.json and would otherwise
    build its index from a half-written file.
    """
    problems = [f"missing key: {k}" for k in REQUIRED_KEYS if k not in run]
    if problems:
        return problems
    if not run["complete"]:
        problems.append("complete is false")
    if not run["configurations"]:
        problems.append("no chunk configurations were evaluated")
    for s in run["configurations"]:
        tag = f"{s.get('chunk_target_tokens')}/{s.get('chunk_overlap_pct')}"
        for key in BOUNDED_METRICS + OPTIONAL_BOUNDED_METRICS:
            v = s.get(key)
            if v is None:
                if key in BOUNDED_METRICS:
                    problems.append(f"{tag}: {key} is missing")
            elif not 0.0 <= v <= 1.0:
                problems.append(f"{tag}: {key}={v} outside [0, 1]")
    sel = run["selected"]
    if not {"chunk_target_tokens", "chunk_overlap_pct"} <= set(sel):
        problems.append("selected is not a (chunk_target_tokens, chunk_overlap_pct) pair")
    elif not any(s["chunk_target_tokens"] == sel["chunk_target_tokens"]
                 and s["chunk_overlap_pct"] == sel["chunk_overlap_pct"]
                 for s in run["configurations"]):
        problems.append("selected configuration is not one of the configurations tested")
    if any(k.lower().startswith("recall@") for s in run["configurations"] for k in s):
        problems.append("Recall@k metric present: section_recall@3 is the only rank-cutoff recall")
    return problems


def is_authoritative(config: dict, limited: bool) -> bool:
    """
    A measurement, as opposed to a smoke run: the production backend, the real
    embedder and the real cross-encoder, over every scored case.
    """
    return (config.get("backend") == "postgres" and config.get("embedder") == "st"
            and config.get("reranker") == "ce" and not limited)


def write_result(run: dict, stamp: str) -> tuple[Path, Path | None]:
    """
    Write the timestamped file always; write a handoff file only for a validated,
    complete run. latest.json is the authoritative handoff to generation evaluation,
    so neither a crashed sweep nor a smoke run may replace it: a completed smoke run
    goes to latest-smoke.json instead. Every write is atomic.
    """
    path = OUT_DIR / f"{stamp}.json"
    payload = json.dumps(run, indent=2, default=str)
    atomic_write_text(path, payload)
    if not run.get("complete"):
        return path, None
    problems = schema_errors(run)
    if problems:
        raise SystemExit("result failed schema validation, no handoff file written:\n  "
                         + "\n  ".join(problems))
    handoff = LATEST if run["authoritative"] else LATEST_SMOKE
    atomic_write_text(handoff, payload)
    return path, handoff


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--sizes", default="256,512,1024",
                    help="chunk_target_tokens values to compare (the agreed sweep is 256,512,1024)")
    ap.add_argument("--overlaps", default=None,
                    help="chunk_overlap_pct values (default: the configured one)")
    ap.add_argument("--limit", type=int, default=None, help="first N scored cases only (smoke)")
    ap.add_argument("--backend", choices=["postgres", "memory"], default="postgres",
                    help="postgres (authoritative, needs --pg-dsn / EVAL_PG_DSN) or memory (smoke)")
    ap.add_argument("--pg-dsn", default=os.environ.get("EVAL_PG_DSN"),
                    help="a DISPOSABLE pgvector database; only eval_retrieval_* schemas are touched")
    ap.add_argument("--keep-schemas", action="store_true",
                    help="leave each configuration's index schema in the database for inspection")
    ap.add_argument("--embedder", choices=["hash", "st"], default=None,
                    help="hash is a SMOKE-TEST option; its numbers are not measurements")
    ap.add_argument("--reranker", choices=["none", "ce"], default=None)
    args = ap.parse_args(argv)

    from flight_delay.config import get_settings, revalidate
    from flight_delay.embeddings import build_embedder, chunk_token_counter_name
    from flight_delay.ingest import PARSER_VERSION
    from flight_delay.retrieval import build_reranker

    if args.backend == "postgres" and not args.pg_dsn:
        raise SystemExit(
            "the authoritative backend needs a disposable pgvector database: set EVAL_PG_DSN or "
            "pass --pg-dsn (RUNBOOK.md, 'Stage 1'). --backend memory runs an offline smoke test.")

    base = get_settings()
    overrides = {k: v for k, v in (("embedder", args.embedder), ("reranker", args.reranker)) if v}
    if overrides.get("embedder") == "hash":
        overrides.setdefault("embedding_dim", 256)
    base = revalidate(base.model_copy(update=overrides))

    sizes = [int(x) for x in args.sizes.split(",")]
    overlaps = [int(x) for x in args.overlaps.split(",")] if args.overlaps else [base.chunk_overlap_pct]
    rows = load_golden_rows()
    items, excluded = split_items(rows, args.limit)

    print(f"{len(items)} scored golden cases ({len(excluded)} excluded); "
          f"sizes {sizes}; overlaps {overlaps}; backend {args.backend}")

    embedder = build_embedder(base)   # loaded once and shared, so sizes differ only by chunking
    reranker = build_reranker(base)

    config = {
        "backend": args.backend,
        "lexical": LEXICAL_POSTGRES if args.backend == "postgres" else LEXICAL_MEMORY,
        **(pg_versions(args.pg_dsn) if args.backend == "postgres" else {}),
        "embedder": base.embedder, "embedding_model": base.embedding_model,
        "embedding_revision": getattr(embedder, "revision", None),
        "embedding_dim": base.embedding_dim,
        "reranker": base.reranker, "reranker_model": base.reranker_model,
        "reranker_max_length": base.reranker_max_length,
        "fuse_rerank_with_dense": base.fuse_rerank_with_dense,
        "sparse_candidates_k": base.sparse_candidates_k,
        "governing_candidates_k": base.governing_candidates_k,
        "structured_query": base.structured_query, "topic_filter": base.topic_filter,
        "max_chunks_per_section": base.max_chunks_per_section,
        "lane_candidates_k": base.lane_candidates_k,
        "scope_lanes": base.scope_lanes,
        "procedural_lanes": base.procedural_lanes,
        "min_airline_sources_legal": base.min_airline_sources_legal,
        "lane_seats": base.lane_seats, "max_lane_seats": base.max_lane_seats,
        "seat_remedy_lanes": base.seat_remedy_lanes,
        "lane_seat_section_diverse": base.lane_seat_section_diverse,
        "max_seats_per_lane": base.max_seats_per_lane,
        "reranker_score_activation": getattr(reranker, "score_activation", None),
        "final_k": base.final_k, "candidates_k": base.candidates_k, "rrf_k": base.rrf_k,
        "use_dense": base.use_dense, "use_sparse": base.use_sparse,
        "hnsw_ef_search": base.hnsw_ef_search,
        "balance_sources": base.balance_sources,
        "min_government_sources": base.min_government_sources,
        "min_airline_sources": base.min_airline_sources,
        "chunk_breadcrumb": base.chunk_breadcrumb,
        "chunk_token_counter": chunk_token_counter_name(base),
        "parser_version": PARSER_VERSION,
        "context_token_budget": base.context_token_budget,
        "llm_context_window": base.llm_context_window,       # the input-assembly budget
        "context_answer_reserve_tokens": base.context_answer_reserve_tokens,
        "max_validation_retries": base.max_validation_retries,
        "query_instruction": base.query_instruction,
        "confidence_gate": "off in Stage 1; its decision is recorded (thresholds uncalibrated)",
        "sizes": sizes, "overlaps": overlaps,
    }
    config_digest = digest(json.dumps({k: v for k, v in config.items() if k not in ("sizes", "overlaps")},
                                      sort_keys=True, default=str))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    summaries: list[dict] = []
    per_case: dict[str, list[dict]] = {}

    authoritative = is_authoritative(config, limited=bool(args.limit))
    if not authoritative:
        print("  NOT AUTHORITATIVE (memory backend, stand-in model or --limit): a completed run is "
              f"written to {display_path(LATEST_SMOKE)}, never {display_path(LATEST)}.")
    digests = {"golden_version": golden_version(), "golden_digest": golden_digest(),
               "corpus_digest": corpus_digest(), "prompt_digest": prompt_digest()}

    def snapshot(complete: bool) -> dict:
        run = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "complete": complete,
            "authoritative": authoritative,
            "stage": "retrieval",
            "config": config,
            "config_digest": config_digest,
            "n_cases": len(items),
            "canonical_case_ids": [i["id"] for i in items],
            "excluded": excluded,
            **digests,
            "configurations": summaries,
            "items": per_case,
        }
        if complete:
            pick = select_configuration(summaries)
            run["selected"] = {"chunk_target_tokens": pick["chunk_target_tokens"],
                               "chunk_overlap_pct": pick["chunk_overlap_pct"]}
            run["selection_rule"] = SELECTION_RULE
        return run

    for overlap in overlaps:
        for size in sizes:
            t0 = time.time()
            summary, case_rows = evaluate_config(
                base, size, overlap, items, embedder, reranker,
                backend=args.backend, dsn=args.pg_dsn, keep_schema=args.keep_schemas)
            summaries.append(summary)
            per_case[f"{size}/{overlap}"] = case_rows
            # Rewritten after every configuration, with complete=false, so an
            # interrupted run keeps its finished work without ever claiming a pick.
            write_result(snapshot(complete=False), stamp)
            print(f"  {size:5} tok /{overlap:3}%  chunks={summary['n_chunks']:5}  "
                  f"evidence_recall={summary['evidence_recall']:.3f}  "
                  f"MRR={summary['mrr']:.3f}  ctx_tok={summary['context_tokens']:.0f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    run = snapshot(complete=True)
    path, latest = write_result(run, stamp)
    print_report(run)
    print(f"\nsaved: {display_path(path)}")
    if latest == LATEST:
        print(f"saved: {display_path(latest)}   (read by evals/generation_eval.py)")
    else:
        print(f"saved: {display_path(latest)}   (smoke handoff; generation_eval.py reads it only "
              f"with --retrieval-result {display_path(latest)} --allow-smoke-selection)")
    sel = run["selected"]
    print(f"\nTo adopt it: set CHUNK_TARGET_TOKENS={sel['chunk_target_tokens']} and "
          f"CHUNK_OVERLAP_PCT={sel['chunk_overlap_pct']} (config.py), then re-index "
          "(scripts/index_corpus.py).")


if __name__ == "__main__":
    main()
