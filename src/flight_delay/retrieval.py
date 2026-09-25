"""
Retrieval: hybrid candidate generation, rank fusion, cross-encoder reranking.

"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .models import Chunk

# ==========================================================================
# Fusion
# ==========================================================================


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    """
    Fuse N ranked lists of (id, score) into one ranked list of (id, rrf_score).

    Input scores are DISCARDED by design — only position matters.
    """
    fused: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (doc_id, _score) in enumerate(lst, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


# ==========================================================================
# Reranking
# ==========================================================================


class NoopReranker:
    """Identity reranker. Used for the ablation baseline and for the hash-embedder
    smoke path where no model download has happened yet."""

    name = "none"

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        return chunks[:top_k]


class CrossEncoderReranker:
    """
    Real cross-encoder. Lazily imported so torch is only required if used.
    """

    name = "ce"

    # The scale rerank_score is on. Recorded by evaluations, so a threshold
    # calibrated on one scale is never silently applied to another.
    score_activation = "identity (raw logits)"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None,
                 max_length: int = 512):
        import torch
        from sentence_transformers import CrossEncoder  # lazy

        # max_length applies to the QUERY + CHUNK pair; longer pairs are truncated
        # (bge-reranker-base/large: 512). Evaluations measure truncation with the
        # model's own tokenizer.
        
        self._load = lambda: CrossEncoder(model_name, max_length=max_length, device=device,
                                          activation_fn=torch.nn.Identity())
       
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        if not chunks:
            return []
        pairs = [(query, c.embed_text) for c in chunks]
        scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)
        for c, s in zip(chunks, scores, strict=True):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: -(c.rerank_score or 0.0))[:top_k]


def _fuse_orders(orders: list[list[Chunk]], k: int) -> list[Chunk]:
    """RRF over orderings of Chunk objects; sets each chunk's score to its fused score."""
    scores: dict[str, float] = {}
    first: dict[str, Chunk] = {}
    for order in orders:
        for rank, ch in enumerate(order, start=1):
            scores[ch.chunk_id] = scores.get(ch.chunk_id, 0.0) + 1.0 / (k + rank)
            first.setdefault(ch.chunk_id, ch)
    out = sorted(first.values(), key=lambda c: -scores[c.chunk_id])
    for ch in out:
        ch.score = scores[ch.chunk_id]
    return out


def build_reranker(settings):
    if settings.reranker == "none":
        return NoopReranker()
    return CrossEncoderReranker(model_name=settings.reranker_model,
                                max_length=settings.reranker_max_length)


# ==========================================================================
# Source balancing
# ==========================================================================


def _binding(chunk: Chunk) -> bool:
    """Government text that can stand for a regime: law, not a regulator's guidance on it."""
    return chunk.source_class == "government" and chunk.doc_type != "guidance"


def balance_by_source_class(
    ranked: list[Chunk],
    top_k: int,
    *,
    min_government: int = 2,
    min_airline: int = 2,
    require_jurisdictions: tuple[str, ...] = (),
    prefer: list[list[Chunk]] | None = None,
    prefer_section_diverse: bool = False,
    prefer_rounds: int = 1,
    prefer_repeat: list[bool] | None = None,
    prefer_max: int = 0,
) -> list[Chunk]:
    """
    Guarantee that the final context contains BOTH regulation and airline text.
    """
    if top_k <= 0 or not ranked:
        return []

    gov = [c for c in ranked if c.source_class == "government"]
    air = [c for c in ranked if c.source_class == "airline"]

    # Never reserve more slots than exist, in the pool or in the budget.
    n_gov = min(min_government, len(gov), top_k)
    n_air = min(min_airline, len(air), top_k - n_gov)

    picked: list[Chunk] = gov[:n_gov] + air[:n_air]
    picked_ids = {c.chunk_id for c in picked}

    # One guaranteed slot per required regime, if the pool can supply it.
    for juris in require_jurisdictions:
        if len(picked) >= top_k:
            break
        # Only a REGULATION satisfies a regime: airline contracts are filed under US,
        # so counting them let UK->US contexts keep no US law at all (Session 17).
        # Nor does a regulator's guidance: the plain-English CAA page out-ranked
        # UK261 and took its seat, leaving live-01 without the Regulation (Session 22).
        if any(_binding(c) and c.jurisdiction == juris for c in picked):
            continue
        best = next(
            (c for c in gov if _binding(c) and c.jurisdiction == juris
             and c.chunk_id not in picked_ids),
            None,
        )
        if best is not None:
            # Mark it: it earned its place from the ROUTE, not from relevance, so it
            # sorts last below and the context budget must not treat it as the
            # weakest thing present (models.Chunk.guaranteed).
            best.guaranteed = True
            picked.append(best)
            picked_ids.add(best.chunk_id)

    # Chunks a lane asked for BY NAME (retrieval: the scope and procedural lanes).
    # Seated after the governing guarantee and before relevance, and marked the same
    # way, because they are here for the same reason: the route and the question
    # decided they belong, and the ranking has no way to know that.
    #
    # `prefer` is one CANDIDATE LIST per lane, not one chunk. With
    # `prefer_section_diverse` OFF a lane offers only its best hit and loses the seat
    # if something already took that chunk. With it ON no seat may repeat a
    # doc_id#section_id already present (two seats went to two chunks of 260.2 on
    # uk-us-14, once because the governing guarantee had taken that section first),
    # and a lane blocked that way falls through to its next candidate rather than
    # lose the seat. `prefer_rounds` > 1 then gives each REPEATABLE list a further seat.
    seated_sections = {(c.doc_id, c.section_id) for c in picked}
    offered: set[str] = set()

    def seat(c: Chunk) -> int:
        c.guaranteed = True
        picked.append(c)
        picked_ids.add(c.chunk_id)
        seated_sections.add((c.doc_id, c.section_id))
        return 1

    def seat_one(candidates: list[Chunk]) -> int:
        for c in candidates:
            if c.chunk_id in offered:
                continue          # another lane already spoke for this chunk
            offered.add(c.chunk_id)
            if not prefer_section_diverse:
                # One offer per lane: if the pool already holds it, the seat is spent.
                return 0 if c.chunk_id in picked_ids else seat(c)
            if c.chunk_id in picked_ids or (c.doc_id, c.section_id) in seated_sections:
                continue
            return seat(c)
        return 0

    seats = 0
    lists = list(prefer or ())
    # Only a lane that can hold more than one premise is offered a further seat: a
    # scope or procedural lane asks for ONE section, so its second-best hit is the
    # same rule restated, while a remedy lane's regime often owes several articles.
    repeat = prefer_repeat if prefer_repeat is not None else [True] * len(lists)
    for round_no in range(max(1, prefer_rounds)):
        for i, candidates in enumerate(lists):
            if len(picked) >= top_k or (prefer_max and seats >= prefer_max):
                break
            if round_no and not repeat[i]:
                continue
            seats += seat_one(candidates)

    for c in ranked:
        if len(picked) >= top_k:
            break
        if c.chunk_id not in picked_ids:
            picked.append(c)
            picked_ids.add(c.chunk_id)

    # Restore the reranker's ordering. build_context relies on best-first.
    order = {c.chunk_id: i for i, c in enumerate(ranked)}
    picked.sort(key=lambda c: order[c.chunk_id])
    return picked[:top_k]


# ==========================================================================
# Topic filter and section cap (Session 22)
# ==========================================================================

# A denied-boarding section is recognised from its OWN headings, not its text:
# "compensation" is everywhere, "denied boarding" in a heading is not. A heading
# that also names a delay or cancellation ("4.4. Right to compensation in the
# event of denied boarding, cancellation, delay at arrival") covers more than
# bumping and is kept.
_DENIED_BOARDING_HEADING_RE = re.compile(
    r"denied[ -]boarding|oversal|oversold|overbook|\bbump(?:ed|ing)?\b", re.IGNORECASE)
_DELAY_OR_CANCEL_RE = re.compile(r"delay|cancel", re.IGNORECASE)
# Documents that are about nothing else (the title says so).
DENIED_BOARDING_DOCS = frozenset({"us-cfr-250-oversales"})


def is_denied_boarding_chunk(chunk: Chunk) -> bool:
    """True for a chunk whose section is about denied boarding and nothing else."""
    if chunk.doc_id in DENIED_BOARDING_DOCS:
        return True
    # breadcrumb = "doc title > heading > sub-heading"; the title is skipped because
    # a document title lists every topic it covers.
    for segment in chunk.breadcrumb.split(" > ")[1:]:
        if _DENIED_BOARDING_HEADING_RE.search(segment) and not _DELAY_OR_CANCEL_RE.search(segment):
            return True
    return False


_CANCELLATION_HEADING_RE = re.compile(r"cancel", re.IGNORECASE)
_DELAY_HEADING_RE = re.compile(r"delay|\blate\b", re.IGNORECASE)


def is_cancellation_only_chunk(chunk: Chunk) -> bool:
    """True for a chunk whose section is about cancellations and not delays."""
    return any(_CANCELLATION_HEADING_RE.search(s) and not _DELAY_HEADING_RE.search(s)
               for s in chunk.breadcrumb.split(" > ")[1:])


# Topic -> test. The pipeline names the topics a question is NOT about
# (pipeline.excluded_topics); their sections leave the pool before reranking.
TOPIC_TESTS = {
    "denied_boarding": is_denied_boarding_chunk,
    # Session 22: a delay question kept the CAA's cancellation-notice tables
    # ("Seven to 14 days' notice") in place of UK261 Art 7 (live-01).
    "cancellation": is_cancellation_only_chunk,
}


def cap_per_section(chunks: list[Chunk], max_per_section: int) -> list[Chunk]:
    """
    Keep at most `max_per_section` chunks of each doc_id#section_id, best first.

    Not one per section: a long section (United Rule 24, a contract's refund rule)
    can hold two separate relevant clauses. Three or more of one section in eight
    slots is what pushed EU/UK Article 7 out beside Article 6.
    """
    if max_per_section <= 0:
        return chunks
    seen: dict[tuple[str, str], int] = {}
    out = []
    for c in chunks:
        key = (c.doc_id, c.section_id)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= max_per_section:
            out.append(c)
    return out


# ==========================================================================
# Hybrid retriever
# ==========================================================================


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    timings_ms: dict[str, float]
    n_dense: int
    n_sparse: int
    n_fused: int
    n_government: int = 0
    n_airline: int = 0
    # The whole candidate pool in reranker order, before source balancing and
    # before the context budget: the ranked list evaluations score.
    ranked: list[Chunk] = field(default_factory=list)


class HybridRetriever:
    """
    Orchestrates: dense + sparse -> RRF -> hydrate -> rerank -> top_k.

    Every stage is switchable from config so the ablation table in RESULTS.md is
    produced by changing settings, not by editing code. If the knobs are not
    exposed here, the ablations do not get run.
    """

    def __init__(self, store, embedder, reranker, settings):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.s = settings

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidates_k: int | None = None,
        flt: dict | None = None,
        exclude_topics: tuple[str, ...] = (),
        lanes: tuple[tuple[str, str], ...] = (),
        min_airline: int | None = None,
    ) -> RetrievalResult:
        """
        `exclude_topics` (names from TOPIC_TESTS; the pipeline passes the topics a
        question is not about when settings.topic_filter is on) removes those
        sections from the pool before reranking.

        `min_airline` overrides settings.min_airline_sources for this turn alone
        (pipeline.airline_quota lowers it for a question no carrier page can answer).
        """
        unknown = set(exclude_topics) - set(TOPIC_TESTS)
        if unknown:
            raise ValueError(f"unknown topics: {sorted(unknown)}")
        excluded = [TOPIC_TESTS[t] for t in exclude_topics]
        top_k = top_k or self.s.final_k
        candidates_k = candidates_k or self.s.candidates_k
        timings: dict[str, float] = {}

        dense: list[tuple[str, float]] = []
        sparse: list[tuple[str, float]] = []

        if self.s.use_dense:
            t0 = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            timings["embed"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dense = self.store.dense_search(qvec, candidates_k, flt)
            timings["dense"] = (time.perf_counter() - t0) * 1000

        sparse_k = getattr(self.s, "sparse_candidates_k", candidates_k)
        if self.s.use_sparse and sparse_k:
            t0 = time.perf_counter()
            sparse = self.store.sparse_search(query, sparse_k, flt)
            timings["sparse"] = (time.perf_counter() - t0) * 1000

        # Candidate pool (Session 17, measured on the golden set at 512 tokens):
        # every dense candidate, then the few best lexical hits dense did not
        # find (exact tokens like "§ 250.5" that embeddings smear). The old pool,
        # an RRF of 50 dense and 50 full-text hits cut to 50, pushed gold dense
        # hits out: section recall of the pool 0.78 vs 0.82 for dense alone, and
        # evidence recall of the final context 0.38 vs 0.45.
        t0 = time.perf_counter()
        if self.s.use_dense and dense:
            seen = {cid for cid, _ in dense}
            fused = list(dense) + [(cid, s) for cid, s in sparse if cid not in seen]
        else:
            lists = [lst for lst in (dense, sparse) if lst]
            fused = reciprocal_rank_fusion(lists, k=self.s.rrf_k) if lists else []
            fused = fused[:candidates_k]
        timings["fuse"] = (time.perf_counter() - t0) * 1000

        # The balancer can only guarantee a governing regime a seat if the pool
        # holds that regime's text. Measured (Session 17): UK->EU and EU->US
        # questions phrased around "US DOT" or a carrier often fused 50 candidates
        # with no UK261/EU261 chunk at all. So each governing regime's own
        # regulations get a small dense search of their own, added to the pool
        # (the reranker still decides their order).
        governing = (flt or {}).get("jurisdiction_governing") or ()
        if isinstance(governing, str):
            governing = [governing]
        extra_k = getattr(self.s, "governing_candidates_k", 0)
        if governing and extra_k and self.s.use_dense:
            t0 = time.perf_counter()
            have = {cid for cid, _ in fused}
            for juris in governing:
                hits = self.store.dense_search(
                    qvec, extra_k, {"jurisdiction": juris, "doc_type": "regulation"})
                for cid, _ in hits:
                    if cid not in have:
                        fused.append((cid, 0.0))
                        have.add(cid)
            timings["governing"] = (time.perf_counter() - t0) * 1000

        # Remedy lanes (Session 23): (jurisdiction, query) searches over that regime's
        # government texts, one per remedy the question needs. Hits join the pool and
        # each lane's order becomes one more vote in the final fusion below.
        lane_orders: list[list[str]] = []
        seat_orders: list[tuple[str, list[str]]] = []
        lane_k = getattr(self.s, "lane_candidates_k", 0)
        if lanes and lane_k and self.s.use_dense:
            t0 = time.perf_counter()
            have = {cid for cid, _ in fused}
            for lane in lanes:
                juris, lane_query, *rest = lane
                kind = rest[0] if rest else "remedy"
                hits = self.store.dense_search(
                    self.embedder.embed_query(lane_query), lane_k,
                    {"jurisdiction": juris, "doc_type": ["regulation", "guidance"]})
                order = [cid for cid, _ in hits]
                lane_orders.append(order)
                # A scope or procedural lane asks for ONE section by name; its best hit
                # is the premise, not a candidate for it (Session 25). A remedy lane
                # asks a broad question, so it keeps its vote and nothing more.
                if kind in ("scope", "procedural", "trigger") or (
                        kind == "remedy" and getattr(self.s, "seat_remedy_lanes", False)):
                    seat_orders.append((kind, order))
                for cid, _ in hits:
                    if cid not in have:
                        fused.append((cid, 0.0))
                        have.add(cid)
            timings["lanes"] = (time.perf_counter() - t0) * 1000

        # Hydrate: one batched fetch rather than N round-trips.
        by_id = self.store.get([cid for cid, _ in fused])
        dense_rank = {cid: i + 1 for i, (cid, _) in enumerate(dense)}
        sparse_rank = {cid: i + 1 for i, (cid, _) in enumerate(sparse)}

        candidates: list[Chunk] = []
        for cid, score in fused:
            ch = by_id.get(cid)
            if ch is None:
                continue
            ch.score = score
            ch.dense_rank = dense_rank.get(cid)
            ch.sparse_rank = sparse_rank.get(cid)
            if any(test(ch) for test in excluded):
                continue
            candidates.append(ch)

        # Rerank the WHOLE candidate pool, not just top_k. The cross-encoder
        # already scores every candidate, so this costs nothing extra, and the
        # balancer below needs to see the lower-ranked chunks to fill its quotas.
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(query, candidates, len(candidates))
        timings["rerank"] = (time.perf_counter() - t0) * 1000

        # Final order: RRF of the dense rank and the cross-encoder rank. Neither
        # alone was best on the golden set (512 tokens, dense top 20, final 8):
        # dense 0.481 evidence recall, bge-reranker-large alone 0.479, fused 0.507
        # (MRR 0.643). Chunks dense never ranked (lexical, governing) get only the
        # reranker's term, so they must earn their place there.
        pool = {c.chunk_id: c for c in reranked}
        lane_votes = [[pool[cid] for cid in order if cid in pool] for order in lane_orders]
        if getattr(self.s, "fuse_rerank_with_dense", False) and dense and \
                any(c.rerank_score is not None for c in reranked):
            reranked = _fuse_orders(
                [[pool[cid] for cid, _ in dense if cid in pool], reranked, *lane_votes], self.s.rrf_k)
        elif any(lane_votes):
            reranked = _fuse_orders([reranked, *lane_votes], self.s.rrf_k)

        reranked = cap_per_section(reranked, getattr(self.s, "max_chunks_per_section", 0))

        if getattr(self.s, "balance_sources", False):
            # Only a regime that GOVERNS the route gets a guaranteed seat.
            # Merely being in scope is not enough: EU261 is retrievable for a
            # US->Paris flight so its scope article can be cited, but reserving
            # it a slot there would push a compensation table it does not owe
            # into the highest-attention position. US is required too: Session 12
            # measured US law crowded out of 10 of 43 contexts (uk-us-06, us-eur-02...)
            # by the non-US regime's slot and airline text.
            require = tuple(governing)

            # One seat per scope/procedural lane, for its best hit that survived the
            # topic filter. Capped, because these are reserved AFTER the governing
            # guarantee and every seat is one less slot decided by relevance.
            prefer: list[list[Chunk]] = []
            prefer_repeat: list[bool] = []
            if getattr(self.s, "lane_seats", False) and seat_orders:
                pool_by_id = {c.chunk_id: c for c in reranked}
                for kind, order in seat_orders:
                    hits = [pool_by_id[cid] for cid in order if cid in pool_by_id]
                    if hits:
                        prefer.append(hits)
                        prefer_repeat.append(kind == "remedy")

            final = balance_by_source_class(
                reranked,
                top_k,
                min_government=self.s.min_government_sources,
                min_airline=self.s.min_airline_sources if min_airline is None else min_airline,
                require_jurisdictions=require,
                prefer=prefer,
                prefer_section_diverse=getattr(self.s, "lane_seat_section_diverse", False),
                prefer_rounds=max(1, getattr(self.s, "max_seats_per_lane", 1)),
                prefer_repeat=prefer_repeat,
                prefer_max=getattr(self.s, "max_lane_seats", 0),
            )
        else:
            final = reranked[:top_k]

        return RetrievalResult(
            chunks=final,
            timings_ms=timings,
            n_dense=len(dense),
            n_sparse=len(sparse),
            n_fused=len(fused),
            n_government=sum(1 for c in final if c.source_class == "government"),
            n_airline=sum(1 for c in final if c.source_class == "airline"),
            ranked=reranked,
        )
