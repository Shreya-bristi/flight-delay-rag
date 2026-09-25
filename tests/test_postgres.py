"""
PostgreSQL integration tests: the SQL the unit tests cannot reach.

"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.models import Chunk  # noqa: E402
from flight_delay.store import PgVectorStore, _filter_sql  # noqa: E402

DSN = os.environ.get("FDR_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="FDR_TEST_PG_DSN not set (needs a disposable Postgres)")

DIM = 8


def _chunk(cid: str, doc: str = "doc-a", text: str = "passengers are entitled to a refund",
           url: str = "https://example.test/a", airline: str = "") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc, doc_title=doc, publisher="p", jurisdiction="US",
                 airline_iata=airline, doc_type="regulation" if not airline else "contract",
                 section_id="s1", breadcrumb=doc, text=text, embed_text=text, source_url=url,
                 effective_date="2026-01-01", token_estimate=10)


def _vecs(n: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def store():
    import psycopg

    schema = f"t_{uuid.uuid4().hex[:12]}"
    s = PgVectorStore(DSN, schema=schema)
    yield s
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def _build(store, chunks, identity=None, required=None):
    staging = store.begin_staging(DIM)
    staging.add(chunks, _vecs(len(chunks)))
    return staging.activate(identity or {"embedding_dim": DIM}, expected_chunks=len(chunks),
                            required_docs=required or sorted({c.doc_id for c in chunks}))


def test_reindexing_preserves_conversations(store):
    _build(store, [_chunk("a1"), _chunk("a2", text="meals after three hours")])
    store.save_message("conv-1", "user", "my flight was cancelled")
    store.save_message("conv-1", "assistant", "you are owed a refund [S1]", [{"marker": "S1"}])

    # A second build, with different chunks, replaces the corpus...
    _build(store, [_chunk("b1", doc="doc-b", text="hotel when overnight")])
    assert store.count() == 1
    assert set(store.get(["b1", "a1"])) == {"b1"}          # obsolete rows are gone
    # ...and the conversation is exactly as it was.
    assert store.history("conv-1") == [("user", "my flight was cancelled"),
                                       ("assistant", "you are owed a refund [S1]")]


def test_failed_validation_leaves_the_live_index_untouched(store):
    _build(store, [_chunk("a1")])
    staging = store.begin_staging(DIM)
    staging.add([_chunk("b1", doc="doc-b")], _vecs(1))
    with pytest.raises(RuntimeError, match="required documents"):
        staging.activate({"embedding_dim": DIM}, expected_chunks=1,
                         required_docs=["doc-b", "doc-missing"])
    staging.abandon()
    assert set(store.get(["a1", "b1"])) == {"a1"}
    assert store.count() == 1


def test_activation_records_identity_and_queries_still_work(store):
    _build(store, [_chunk("a1"), _chunk("u1", doc="ua", airline="UA", text="United rebooks")],
           identity={"embedding_dim": DIM, "corpus_digest": "abc"})
    assert store.active_identity()["corpus_digest"] == "abc"
    # The renamed indexes and the generated tsvector column survive the swap.
    assert [cid for cid, _ in store.sparse_search("rebooks", 5)] == ["u1"]
    assert len(store.dense_search(_vecs(1, seed=3)[0], 5)) == 2
    # Mirror-image filters still behave after activation.
    got = {cid for cid, _ in store.dense_search(_vecs(1)[0], 5, flt={"airline_scope": "DL"})}
    assert got == {"a1"}


def test_upsert_updates_changed_metadata_for_unchanged_text(store):
    """chunk_id hashes text only; a corrected source URL used to be silently ignored."""
    _build(store, [_chunk("a1", url="https://old.test")])
    store.upsert([_chunk("a1", url="https://new.test")], _vecs(1, seed=9))
    assert store.get(["a1"])["a1"].source_url == "https://new.test"


def test_dimension_mismatch_is_refused(store):
    _build(store, [_chunk("a1")])
    with pytest.raises(RuntimeError, match="vector\\(8\\)"):
        store.ensure_schema(1024)


def test_rebuild_after_an_abandoned_staging_table(store):
    store.begin_staging(DIM).add([_chunk("x1")], _vecs(1))   # crashed build left behind
    assert _build(store, [_chunk("a1")]) == 1
    assert set(store.get(["x1", "a1"])) == {"a1"}


def test_conversation_state_round_trips_and_survives_reindexing(store):
    _build(store, [_chunk("a1")])
    store.save_message("conv-s", "user", "My Delta flight from Paris was cancelled")
    state = {"version": 1, "airline": "DL", "flt": {"jurisdiction_governing": ["EU"]}}
    store.save_state("conv-s", state)
    _build(store, [_chunk("b1", doc="doc-b")])
    assert store.load_state("conv-s") == state
    assert store.load_state("missing") is None


def test_quota_reservation_is_atomic_across_connections(store):
    from concurrent.futures import ThreadPoolExecutor

    store.ensure_chat_schema()
    with ThreadPoolExecutor(max_workers=8) as pool:
        granted = sum(pool.map(lambda _: store.quota_try_reserve("2026-09", 25), range(60)))
    assert granted == 25 and store.quota_used("2026-09") == 25
    assert store.quota_try_reserve("2026-10", 0) is False


def test_shared_rate_limit_counts_across_connections(store):
    store.ensure_chat_schema()
    other = PgVectorStore(DSN, schema=store.schema)     # a second replica, same database
    # An hour-long window, so the test cannot straddle a window boundary.
    results = [s.rate_limit_hit("203.0.113.9", 3, 3600) for s in (store, other, store, other)]
    assert results == [False, False, False, True]
    assert other.rate_limit_hit("198.51.100.1", 3, 3600) is False


def test_sparse_search_answers_natural_language_questions(store):
    """Regression: websearch_to_tsquery ANDed every term, so route and flight words the
    clause never mentions ("UA57", "Paris", "Newark") made the lexical lane return nothing."""
    _build(store, [
        _chunk("c1", text="If a flight is cancelled the passenger is entitled to a refund of the ticket price"),
        _chunk("c2", doc="doc-b", text="Pets must travel in an approved kennel"),
    ])
    question = "My United flight UA57 from Paris to Newark was cancelled - am I owed a refund?"
    got = [cid for cid, _ in store.sparse_search(question, 5)]
    assert got and got[0] == "c1"
    assert "c2" not in got
    # Punctuation or stopwords alone neither raise nor match everything.
    assert store.sparse_search("?? to the !!", 5) == []


def test_dense_search_fills_k_when_the_hnsw_index_is_used(store):
    """pgvector's default ef_search (40) returned ~30 of 50 rows under route filters
    whenever the planner chose the HNSW index; dense_search raises it per query."""
    import psycopg

    chunks = [_chunk(f"r{i}", doc=f"d{i % 3}", airline="UA" if i % 2 else "",
                     text=f"clause number {i}") for i in range(240)]
    staging = store.begin_staging(DIM)
    staging.add(chunks, _vecs(len(chunks), seed=4))
    staging.activate({"embedding_dim": DIM}, expected_chunks=len(chunks),
                     required_docs=["d0", "d1", "d2"])

    forced = PgVectorStore(DSN, schema=store.schema)
    options = (f"-c search_path={store.schema},public "
               "-c enable_seqscan=off -c enable_bitmapscan=off")
    forced._conn = lambda: psycopg.connect(DSN, autocommit=True, options=options)
    params: list = []
    where = _filter_sql({"airline_scope": "DL"}, params)       # production's filter shape
    with forced._conn() as c:
        c.execute("ANALYZE chunks")
        plan = " ".join(r[0] for r in c.execute(
            f"EXPLAIN SELECT chunk_id FROM chunks {where} ORDER BY embedding <=> %s::vector LIMIT 50",
            [*params, "[" + ",".join(["0.1"] * DIM) + "]"]))
    assert "chunks_embedding_hnsw" in plan
    got = forced.dense_search(_vecs(1, seed=5)[0], 50, flt={"airline_scope": "DL"})
    assert len(got) == 50            # 120 regulation rows qualify; all 50 requested come back
