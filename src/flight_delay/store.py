"""
Storage: chunks, their vectors, and their full-text index.

"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from typing import Protocol

import numpy as np

from .embeddings import tokenize
from .models import Chunk


class VectorStore(Protocol):
    def ensure_schema(self, dim: int) -> None: ...
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int: ...
    def dense_search(self, qvec: np.ndarray, k: int, flt: dict | None = None) -> list[tuple[str, float]]: ...
    def sparse_search(self, query: str, k: int, flt: dict | None = None) -> list[tuple[str, float]]: ...
    def get(self, chunk_ids: list[str]) -> dict[str, Chunk]: ...
    def count(self) -> int: ...


# ==========================================================================
# Postgres
# ==========================================================================

_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

# One corpus table definition, instantiated under two names: `chunks` (the live
# index every query reads) and `chunks_staging` (a build in progress). Index names
# carry the table name so both can exist at once; activation renames them.
_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    doc_title       TEXT NOT NULL,
    publisher       TEXT,
    jurisdiction    TEXT,
    airline_iata    TEXT,
    doc_type        TEXT,
    section_id      TEXT,
    breadcrumb      TEXT,
    text            TEXT NOT NULL,
    embed_text      TEXT NOT NULL,
    source_url      TEXT,
    effective_date  TEXT,
    token_estimate  INTEGER,
    embedding       vector({dim}),
    -- A GENERATED column means the search vector can never drift out of sync
    -- with the text. Maintaining it in application code is a bug waiting to
    -- happen the first time someone updates a row from psql.
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', embed_text)) STORED
);

-- HNSW rather than IVFFlat: HNSW needs no training step, handles incremental
-- inserts gracefully, and gives better recall at the same latency. IVFFlat's
-- advantage is lower build memory, which does not bind us at this scale.
-- Construction parameters are pgvector's defaults; defaults are not a defect,
-- and filtered search behaviour is measured by evals/retrieval_eval.py.
CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
    ON {table} USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS {table}_tsv_gin ON {table} USING GIN (tsv);
CREATE INDEX IF NOT EXISTS {table}_jurisdiction ON {table} (jurisdiction);
CREATE INDEX IF NOT EXISTS {table}_doc ON {table} (doc_id);
CREATE INDEX IF NOT EXISTS {table}_airline ON {table} (airline_iata);
CREATE INDEX IF NOT EXISTS {table}_doc_type ON {table} (doc_type);
"""

_INDEX_SUFFIXES = ("pkey", "embedding_hnsw", "tsv_gin", "jurisdiction", "doc", "airline", "doc_type")

# Chat history for the API. Kept in the same database as the vectors so a
# conversation and the evidence it cited can be read in one transaction - but in
# tables that no corpus operation ever drops, truncates or rewrites.
_CHAT_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    citations       JSONB,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conv ON messages (conversation_id, created_at);

-- The route and carrier a conversation has settled, so an ordinary follow-up
-- ("what about a hotel?") keeps the filters the first answer used.
CREATE TABLE IF NOT EXISTS conversation_state (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    state           JSONB NOT NULL,
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- AirLabs calls this month, shared by every replica (tools.StoreQuotaCounter).
CREATE TABLE IF NOT EXISTS airlabs_quota (
    period  TEXT PRIMARY KEY,
    used    INTEGER NOT NULL
);

-- Fixed-window request counters for /ask, shared by every replica.
CREATE TABLE IF NOT EXISTS rate_limits (
    bucket      TEXT PRIMARY KEY,
    hits        INTEGER NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL
);
"""

# What the live `chunks` table was built from: one row per activation, newest =
# live. The API refuses readiness when it disagrees with the serving settings.
_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_builds (
    id            BIGSERIAL PRIMARY KEY,
    activated_at  TIMESTAMPTZ DEFAULT now(),
    n_chunks      INTEGER NOT NULL,
    identity      JSONB NOT NULL
);
"""

# Identity fields that must equal the serving configuration.
IDENTITY_SETTINGS = (
    "embedder", "embedding_model", "embedding_dim", "chunk_target_tokens",
    "chunk_overlap_pct", "chunk_breadcrumb",
)


def identity_mismatches(identity: dict | None, settings,
                        parser_version: str | None = None) -> list[str]:
    """
    Why an index built with `identity` must not serve under `settings` (empty = compatible).

    Query vectors from one embedding model searched against document vectors from
    another return confident nonsense rather than an error, so this is checked, not assumed.
    """
    if not identity:
        return ["the index has no recorded build identity; re-index with scripts/index_corpus.py"]
    out = []
    for key in IDENTITY_SETTINGS:
        if identity.get(key) != getattr(settings, key):
            out.append(f"{key}: index built with {identity.get(key)!r}, "
                       f"settings say {getattr(settings, key)!r}")
    if parser_version is not None and identity.get("parser_version") != parser_version:
        out.append(f"parser_version: index built with {identity.get('parser_version')!r}, "
                   f"code is {parser_version!r}")
    return out


def _vec_literal(v: np.ndarray) -> str:
    """pgvector accepts its input as a bracketed string literal."""
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


def _filter_sql(flt: dict | None, params: list) -> str:
    """
    Build an optional WHERE fragment.

    Filtering BEFORE the ANN search is the single highest-leverage retrieval
    improvement available here: restricting to one jurisdiction can cut the
    candidate pool several-fold, which raises effective recall more than any
    embedding model upgrade we could make.
    """
    if not flt:
        return ""
    clauses = []
    for col in ("jurisdiction", "doc_id", "publisher", "airline_iata", "doc_type"):
        if flt.get(col):
            vals = flt[col] if isinstance(flt[col], list) else [flt[col]]
            clauses.append(f"{col} = ANY(%s)")
            params.append(vals)

 
    scope = flt.get("airline_scope")
    if scope:
        vals = scope if isinstance(scope, list) else [scope]
        clauses.append("(airline_iata = ANY(%s) OR airline_iata = '' OR airline_iata IS NULL)")
        params.append(vals)

    # "Regulations from these jurisdictions, plus all airline documents." The
    # mirror image of the above: airline contracts are filed under US but govern
    # the carrier's conduct everywhere it flies, so scoping them by jurisdiction
    # would drop the carrier's own commitments from every Europe question.
    # An explicit EMPTY scope means no regime applies (e.g. DXB -> DEL, both
    # endpoints outside US/EU/UK): keep airline documents, drop every regulation.
    # It must not read as "no filter", which would put all three regimes' law in play.
    jscope = flt.get("jurisdiction_scope")
    if jscope:
        vals = jscope if isinstance(jscope, list) else [jscope]
        clauses.append("(jurisdiction = ANY(%s) OR (airline_iata <> '' AND airline_iata IS NOT NULL))")
        params.append(vals)
    elif jscope is not None:
        clauses.append("(airline_iata <> '' AND airline_iata IS NOT NULL)")

    return (" WHERE " + " AND ".join(clauses)) if clauses else ""


class PgVectorStore:
    """
    pgvector-backed store.

    """

    def __init__(self, dsn: str, schema: str | None = None, hnsw_ef_search: int = 100) -> None:
        import psycopg  
        if schema is not None and not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", schema):
            raise ValueError(f"invalid schema name: {schema!r}")
        if hnsw_ef_search < 1:
            raise ValueError("hnsw_ef_search must be positive")
        self._psycopg = psycopg
        self.dsn = dsn
        self.schema = schema
        self.hnsw_ef_search = hnsw_ef_search

    def _conn(self):
        if self.schema:
            return self._psycopg.connect(
                self.dsn, autocommit=True, options=f"-c search_path={self.schema},public")
        return self._psycopg.connect(self.dsn, autocommit=True)

    # -- schema --------------------------------------------------------------
    @staticmethod
    def _table_dim(cur, table: str) -> int | None:
        cur.execute(
            "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
            "WHERE a.attrelid = to_regclass(%s) AND a.attname = 'embedding' "
            "AND NOT a.attisdropped",
            [table],
        )
        row = cur.fetchone()
        m = re.fullmatch(r"vector\((\d+)\)", row[0]) if row and row[0] else None
        return int(m.group(1)) if m else None

    def _base_schema(self, cur) -> None:
        if self.schema:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
        cur.execute(_EXTENSION)
        cur.execute(_CHAT_SCHEMA)
        cur.execute(_META_SCHEMA)

    def ensure_schema(self, dim: int) -> None:
        """
        Create whatever is missing. Never drops anything, and refuses an existing
        live index built for a different embedding dimension.
        """
        with self._conn() as c, c.cursor() as cur:
            self._base_schema(cur)
            existing = self._table_dim(cur, "chunks")
            if existing is not None and existing != dim:
                raise RuntimeError(
                    f"the live chunks table stores vector({existing}) but the embedder produces "
                    f"{dim} dimensions. Re-index with scripts/index_corpus.py to replace the "
                    "corpus; conversations are not affected.")
            cur.execute(_CHUNKS_TABLE.format(table="chunks", dim=dim))

    def ensure_chat_schema(self) -> None:
        """Conversation tables only (API startup): serving must not depend on indexing."""
        with self._conn() as c, c.cursor() as cur:
            if self.schema:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            cur.execute(_CHAT_SCHEMA)

    # -- corpus replacement --------------------------------------------------
    def begin_staging(self, dim: int) -> StagingIndex:
        """
        Start a fresh corpus build in `chunks_staging`, beside the live index.

        Queries keep reading `chunks` for the whole build; nothing a passenger sees
        changes until `StagingIndex.activate()` swaps the tables.
        """
        with self._conn() as c, c.cursor() as cur:
            self._base_schema(cur)
            cur.execute("DROP TABLE IF EXISTS chunks_staging")   # an abandoned earlier build
            cur.execute(_CHUNKS_TABLE.format(table="chunks_staging", dim=dim))
        return StagingIndex(self, dim)

    def active_identity(self) -> dict | None:
        """The identity recorded by the most recent activation, or None."""
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT to_regclass('index_builds')")
            if cur.fetchone()[0] is None:
                return None
            cur.execute("SELECT identity FROM index_builds ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            return row[0] if row else None

    def _insert(self, table: str, chunks: list[Chunk], vectors: np.ndarray) -> int:
        if not chunks:
            return 0
        rows = [
            (
                ch.chunk_id, ch.doc_id, ch.doc_title, ch.publisher, ch.jurisdiction, ch.airline_iata,
                ch.doc_type, ch.section_id, ch.breadcrumb, ch.text, ch.embed_text, ch.source_url,
                ch.effective_date, ch.token_estimate, _vec_literal(vectors[i]),
            )
            for i, ch in enumerate(chunks)
        ]
        # DO UPDATE, not DO NOTHING: chunk_id hashes the section text only, so the
        # same text with corrected metadata (a source URL, a doc_type) or a new
        # embedding has the same id and must overwrite the stale row.
        sql = f"""
            INSERT INTO {table} (chunk_id, doc_id, doc_title, publisher, jurisdiction,
                airline_iata, doc_type, section_id, breadcrumb, text, embed_text, source_url,
                effective_date, token_estimate, embedding)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id, doc_title = EXCLUDED.doc_title,
                publisher = EXCLUDED.publisher, jurisdiction = EXCLUDED.jurisdiction,
                airline_iata = EXCLUDED.airline_iata, doc_type = EXCLUDED.doc_type,
                section_id = EXCLUDED.section_id, breadcrumb = EXCLUDED.breadcrumb,
                text = EXCLUDED.text, embed_text = EXCLUDED.embed_text,
                source_url = EXCLUDED.source_url, effective_date = EXCLUDED.effective_date,
                token_estimate = EXCLUDED.token_estimate, embedding = EXCLUDED.embedding
        """
        with self._conn() as c, c.cursor() as cur:
            cur.executemany(sql, rows)
        return len(rows)

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Insert or update rows in the LIVE index. Corpus builds go through begin_staging()."""
        return self._insert("chunks", chunks, vectors)

    def dense_search(self, qvec, k, flt=None):
        params: list = []
        where = _filter_sql(flt, params)
        # `<=>` is cosine DISTANCE (0 = identical). We return similarity so that
        # higher is better everywhere in this codebase, which avoids a whole
        # class of sign-flip bugs in the fusion step.
        sql = f"""
            SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score
            FROM chunks {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        lit = _vec_literal(qvec)
        with self._conn() as c, c.transaction(), c.cursor() as cur:
            # HNSW search settings
            cur.execute("SELECT set_config('hnsw.ef_search', %s, true), "
                        "set_config('hnsw.iterative_scan', 'strict_order', true)",
                        [str(max(self.hnsw_ef_search, k))])
            cur.execute(sql, [lit, *params, lit, k])
            return [(r[0], float(r[1])) for r in cur.fetchall()]

    def sparse_search(self, query, k, flt=None):
        params: list = []
        where = _filter_sql(flt, params)
        joiner = " AND " if where else " WHERE "
        # PostgreSQL full-text search
        sql = f"""
            SELECT chunk_id, ts_rank_cd(tsv, qq.q, 1) AS score
            FROM chunks,
                 (SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')::tsquery AS q) qq
            {where}{joiner} tsv @@ qq.q
            ORDER BY score DESC
            LIMIT %s
        """
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, [query, *params, k])
            return [(r[0], float(r[1])) for r in cur.fetchall()]

    def get(self, chunk_ids):
        if not chunk_ids:
            return {}
        sql = """
            SELECT chunk_id, doc_id, doc_title, publisher, jurisdiction, airline_iata, doc_type,
                   section_id, breadcrumb, text, embed_text, source_url, effective_date,
                   token_estimate
            FROM chunks WHERE chunk_id = ANY(%s)
        """
        with self._conn() as c, c.cursor() as cur:
            cur.execute(sql, [list(chunk_ids)])
            out = {}
            for r in cur.fetchall():
                out[r[0]] = Chunk(
                    chunk_id=r[0], doc_id=r[1], doc_title=r[2], publisher=r[3],
                    jurisdiction=r[4], airline_iata=r[5] or "", doc_type=r[6] or "",
                    section_id=r[7], breadcrumb=r[8], text=r[9],
                    embed_text=r[10], source_url=r[11], effective_date=r[12],
                    token_estimate=r[13] or 0,
                )
            return out

    def count(self) -> int:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT to_regclass('chunks')")
            if cur.fetchone()[0] is None:
                return 0
            cur.execute("SELECT count(*) FROM chunks")
            return int(cur.fetchone()[0])

    # -- chat history ------------------------------------------------------
    def save_message(self, conv_id, role, content, citations=None):
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (id) VALUES (%s) ON CONFLICT DO NOTHING", [conv_id]
            )
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content, citations) "
                "VALUES (%s,%s,%s,%s)",
                [conv_id, role, content, json.dumps(citations or [])],
            )

    def history(self, conv_id, limit=6):
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM messages WHERE conversation_id=%s "
                "ORDER BY created_at DESC, id DESC LIMIT %s",
                [conv_id, limit],
            )
            return list(reversed([(r[0], r[1]) for r in cur.fetchall()]))

    # -- conversation state ------------------------------------------------
    def load_state(self, conv_id) -> dict | None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT state FROM conversation_state WHERE conversation_id=%s", [conv_id])
            row = cur.fetchone()
            return row[0] if row else None

    def save_state(self, conv_id, state: dict) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (id) VALUES (%s) ON CONFLICT DO NOTHING", [conv_id])
            cur.execute(
                "INSERT INTO conversation_state (conversation_id, state) VALUES (%s, %s) "
                "ON CONFLICT (conversation_id) DO UPDATE SET state = EXCLUDED.state, "
                "updated_at = now()",
                [conv_id, json.dumps(state, default=str)],
            )

    # -- shared counters -----------------------------------------------------
    def quota_used(self, period: str) -> int:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT used FROM airlabs_quota WHERE period=%s", [period])
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def quota_try_reserve(self, period: str, limit: int) -> bool:
        """Count one call if the month's total stays <= limit. One statement: atomic."""
        if limit < 1:
            return False
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO airlabs_quota (period, used) VALUES (%s, 1) "
                "ON CONFLICT (period) DO UPDATE SET used = airlabs_quota.used + 1 "
                "WHERE airlabs_quota.used + 1 <= %s RETURNING used",
                [period, limit],
            )
            return cur.fetchone() is not None

    def rate_limit_hit(self, key: str, limit: int, window_s: int) -> bool:
        """Record one request for `key`; True when it exceeds `limit` in the current window."""
        window = int(time.time() // window_s)
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO rate_limits (bucket, hits, expires_at) "
                "VALUES (%s, 1, now() + make_interval(secs => %s)) "
                "ON CONFLICT (bucket) DO UPDATE SET hits = rate_limits.hits + 1 RETURNING hits",
                [f"{key}:{window_s}:{window}", window_s * 2],
            )
            hits = cur.fetchone()[0]
            if window % 10 == 0 and hits == 1:
                cur.execute("DELETE FROM rate_limits WHERE expires_at < now()")
            return hits > limit


class StagingIndex:
    """
    A corpus build in `chunks_staging`: fill it, then activate it.

    Activation validates the staged rows and then swaps them in within ONE
    transaction (Postgres DDL is transactional): the previous corpus table is
    dropped and the staging table and its indexes are renamed into place. Readers
    see the whole old index or the whole new one, never a half-loaded table, and
    obsolete rows leave with the old table. Conversations are separate tables and
    are never touched.
    """

    def __init__(self, store: PgVectorStore, dim: int) -> None:
        self.store = store
        self.dim = dim

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        return self.store._insert("chunks_staging", chunks, vectors)

    def doc_counts(self) -> dict[str, int]:
        with self.store._conn() as c, c.cursor() as cur:
            cur.execute("SELECT doc_id, count(*) FROM chunks_staging GROUP BY doc_id")
            return {r[0]: int(r[1]) for r in cur.fetchall()}

    def activate(self, identity: dict, expected_chunks: int, required_docs: list[str]) -> int:
        """Validate the staged corpus and make it live. Raises, changing nothing, on failure."""
        counts = self.doc_counts()
        n = sum(counts.values())
        problems = []
        if n != expected_chunks:
            problems.append(f"staging holds {n} chunks but the build produced {expected_chunks}")
        missing = [d for d in required_docs if not counts.get(d)]
        if missing:
            problems.append(f"no chunks for required documents: {', '.join(missing)}")
        if problems:
            raise RuntimeError("staged index failed validation; the live index was not changed: "
                               + "; ".join(problems))
        with self.store._conn() as c, c.transaction(), c.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS chunks")
            cur.execute("ALTER TABLE chunks_staging RENAME TO chunks")
            for suffix in _INDEX_SUFFIXES:
                cur.execute(f"ALTER INDEX IF EXISTS chunks_staging_{suffix} RENAME TO chunks_{suffix}")
            cur.execute("INSERT INTO index_builds (n_chunks, identity) VALUES (%s, %s)",
                        [n, json.dumps(identity, sort_keys=True, default=str)])
        return n

    def abandon(self) -> None:
        with self.store._conn() as c, c.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS chunks_staging")


# ==========================================================================
# In-memory
# ==========================================================================


class MemoryStore:
    """
    Brute-force numpy search plus a real Okapi BM25. No Postgres required.

    Brute force is exact, which is useful: when a retrieval test fails you know
    it is your ranking logic, not an approximate index's recall. At 40k chunks a
    full matmul is ~30ms, so this is also perfectly usable for small corpora.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.ids: list[str] = []
        self.matrix: np.ndarray | None = None
        self.k1, self.b = k1, b
        self._docs: list[list[str]] = []
        self._df: Counter = Counter()
        self._avg_len = 0.0

    def ensure_schema(self, dim: int) -> None:  # noqa: ARG002 - protocol parity
        return None

    def upsert(self, chunks, vectors):
        for i, ch in enumerate(chunks):
            if ch.chunk_id in self.chunks:
                continue
            self.chunks[ch.chunk_id] = ch
            self.ids.append(ch.chunk_id)
            v = vectors[i].reshape(1, -1)
            self.matrix = v if self.matrix is None else np.vstack([self.matrix, v])
            toks = tokenize(ch.embed_text)
            self._docs.append(toks)
            for t in set(toks):
                self._df[t] += 1
        if self._docs:
            self._avg_len = sum(len(d) for d in self._docs) / len(self._docs)
        return len(chunks)

    def _passes(self, ch: Chunk, flt: dict | None) -> bool:
        if not flt:
            return True
        for col in ("jurisdiction", "doc_id", "publisher", "airline_iata", "doc_type"):
            want = flt.get(col)
            if want:
                vals = want if isinstance(want, list) else [want]
                if getattr(ch, col) not in vals:
                    return False

        scope = flt.get("airline_scope")
        if scope:
            vals = scope if isinstance(scope, list) else [scope]
            if ch.airline_iata and ch.airline_iata not in vals:
                return False

        jscope = flt.get("jurisdiction_scope")
        if jscope is not None:   # [] = no regime applies: regulations are all excluded
            vals = jscope if isinstance(jscope, list) else [jscope]
            if not ch.airline_iata and ch.jurisdiction not in vals:
                return False
        return True

    def dense_search(self, qvec, k, flt=None):
        if self.matrix is None:
            return []
        # Vectors are unit-normalised, so a dot product IS cosine similarity.
        sims = self.matrix @ qvec
        order = np.argsort(-sims)
        out = []
        for i in order:
            cid = self.ids[i]
            if self._passes(self.chunks[cid], flt):
                out.append((cid, float(sims[i])))
                if len(out) >= k:
                    break
        return out

    def sparse_search(self, query, k, flt=None):
        n = len(self._docs)
        if n == 0:
            return []
        q = tokenize(query)
        scores: list[tuple[str, float]] = []
        for i, doc in enumerate(self._docs):
            cid = self.ids[i]
            if not self._passes(self.chunks[cid], flt):
                continue
            tf = Counter(doc)
            dl = len(doc)
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                df = self._df[term]
                # Okapi BM25 IDF with the +0.5 smoothing that keeps it positive
                # for terms appearing in more than half the corpus.
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                f = tf[term]
                s += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / max(self._avg_len, 1e-9))
                )
            if s > 0:
                scores.append((cid, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def get(self, chunk_ids):
        return {cid: self.chunks[cid] for cid in chunk_ids if cid in self.chunks}

    def count(self):
        return len(self.chunks)

    # chat history: not persisted in the memory store
    def save_message(self, *a, **kw):
        return None

    def history(self, *a, **kw):
        return []

    def load_state(self, conv_id):  # noqa: ARG002
        return None

    def save_state(self, conv_id, state):  # noqa: ARG002
        return None
