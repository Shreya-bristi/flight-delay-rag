#!/usr/bin/env python3
"""
Build and activate the pgvector corpus index.

Usage:
    python scripts/index_corpus.py
    python scripts/index_corpus.py --reset
    python scripts/index_corpus.py --dry-run

Each build is staged and validated before replacing the live index.
The CORPUS manifest is the source of truth for indexed documents and metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
# generated index metadata stays outside the curated source corpus
MANIFEST_OUT = ROOT / "build" / "index-manifest.json"

# --------------------------------------------------------------------------
# Corpus manifest
# keep source URLs and effective dates aligned with data/metadata.xlsx.
#
# --------------------------------------------------------------------------

CORPUS = [
    # ---------------- Government regulations  ----------------
    {
        "doc_id": "us-cfr-250-oversales",
        "path": "government_mandates/US/14_CFR_Part250_oversales.pdf",
        "doc_title": "14 CFR Part 250 — Oversales (Denied Boarding Compensation)",
        "publisher": "US DOT",
        "jurisdiction": "US",
        "doc_type": "regulation",
        "source_url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-250",
        "effective_date": "2026-09-04",
        "pdf_reading_order": "sorted",
    },
    {
        "doc_id": "us-cfr-259-protections",
        "path": "government_mandates/US/14_CFR_Part259_protection.pdf",
        "doc_title": "14 CFR Part 259 — Enhanced Protections for Airline Passengers",
        "publisher": "US DOT",
        "jurisdiction": "US",
        "doc_type": "regulation",
        "source_url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-259",
        "effective_date": "2026-09-04",
        "pdf_reading_order": "sorted",
    },
    {
        "doc_id": "us-cfr-260-refunds",
        "path": "government_mandates/US/14_CFR_Part260_refunds.pdf",
        "doc_title": "14 CFR Part 260 — Refunds for Fare and Ancillary Service Fees",
        "publisher": "US DOT",
        "jurisdiction": "US",
        "doc_type": "regulation",
        "source_url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260",
        "effective_date": "2026-09-04",
        "pdf_reading_order": "sorted",
    },
    {
        "doc_id": "us-dot-refunds-qa",
        "path": "government_mandates/US/dot_refunds_qa.md",
        "doc_title": "US DOT — Airline Refunds Q&A",
        "publisher": "US DOT",
        "jurisdiction": "US",
        "doc_type": "regulation",
        "source_url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
        "effective_date": "2026-09-01",
    },
    {
        "doc_id": "eu-261-2004",
        "path": "government_mandates/EU/EU261_2004.pdf",
        "doc_title": "Regulation (EC) No 261/2004 — Air Passenger Rights",
        "publisher": "European Parliament and Council",
        "jurisdiction": "EU",
        "doc_type": "regulation",
        "source_url": "https://eur-lex.europa.eu/eli/reg/2004/261/oj/eng",
        "effective_date": "2005-02-17",
    },
    # official Commission guidance supplements the binding EU261 regulation
    {
        "doc_id": "eu-261-guidelines",
        "path": "government_mandates/EU/EUR-lex.pdf",
        "doc_title": "European Commission — Interpretative Guidelines on Regulation (EC) No 261/2004 (C/2024/5687)",
        "publisher": "European Commission",
        "jurisdiction": "EU",
        "doc_type": "guidance",
        "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ%3AC_202405687",
        "effective_date": "2024-09-25",
        "pdf_footnotes": "after_page_number",
    },
    {
        "doc_id": "uk-261",
        "path": "government_mandates/UK/UK261.pdf",
        "doc_title": "UK 261 — Retained Air Passenger Rights Regulation",
        "publisher": "UK Government (retained EU law)",
        "jurisdiction": "UK",
        "doc_type": "regulation",
        "source_url": "https://www.legislation.gov.uk/eur/2004/261/contents",
        # Indexed copy includes amendments effective 2023-12-14
        "effective_date": "2023-12-14",
    },
    # UK regulator guidance supplements the binding UK261 regulation
    {
        "doc_id": "uk-caa-guidance",
        "path": "government_mandates/UK/uk_caa_delays.md",
        "doc_title": "UK Civil Aviation Authority — Delays, Cancellations and Denied Boarding",
        "publisher": "UK Civil Aviation Authority",
        "jurisdiction": "UK",
        "doc_type": "guidance",
        "source_url": "https://www.caa.co.uk/air-passengers/travel-problems-and-rights/flight-delays-and-cancellations/delays/",
        "effective_date": "2026-09-19",
    },

    # ---------------- American Airlines ----------------
    {
        "doc_id": "aa-conditions-of-carriage",
        "path": "airline_contracts/american_airlines/conditions_of_carriage_AA.md",
        "doc_title": "American Airlines — Conditions of Carriage",
        "publisher": "American Airlines",
        "jurisdiction": "US",
        "airline_iata": "AA",
        "doc_type": "contract",
        "source_url": "https://www.aa.com/web/i18n/customer-service/support/conditions-of-carriage.html",
        "effective_date": "2026-08-13",
    },
    {
        "doc_id": "aa-customer-service-plan",
        "path": "airline_contracts/american_airlines/customer_service_plan_AA.md",
        "doc_title": "American Airlines — Customer Service Plan",
        "publisher": "American Airlines",
        "jurisdiction": "US",
        "airline_iata": "AA",
        "doc_type": "service_plan",
        "source_url": "https://www.aa.com/web/i18n/customer-service/support/customer-service-plan.html",
        "effective_date": "2026-08-26",
    },

    # ---------------- Delta Air Lines ----------------
    {
        "doc_id": "dl-contract-domestic",
        "path": "airline_contracts/delta/Delta-domestic.pdf",
        "doc_title": "Delta Air Lines — Contract of Carriage (Domestic)",
        "publisher": "Delta Air Lines",
        "jurisdiction": "US",
        "airline_iata": "DL",
        "doc_type": "contract",
        "source_url": "https://www.delta.com/us/en/legal/contract-of-carriage-dgr",
        "effective_date": "2026-08-18",
    },
    {
        "doc_id": "dl-contract-international",
        "path": "airline_contracts/delta/Delta-International.pdf",
        "doc_title": "Delta Air Lines — Contract of Carriage (International)",
        "publisher": "Delta Air Lines",
        "jurisdiction": "US",
        "airline_iata": "DL",
        "doc_type": "contract",
        "source_url": "https://www.delta.com/us/en/legal/contract-of-carriage-igr",
        "effective_date": "2026-08-18",
    },
    {
        "doc_id": "dl-delay-cancel-policy",
        "path": "airline_contracts/delta/delayed_and_canceled_flight_policy_delta.txt",
        "doc_title": "Delta Air Lines — Delayed or Canceled Flight Policy",
        "publisher": "Delta Air Lines",
        "jurisdiction": "US",
        "airline_iata": "DL",
        "doc_type": "policy",
        "source_url": "https://www.delta.com/us/en/change-cancel/delayed-or-canceled-flight",
        "effective_date": "2024-10-01",
    },

    # ---------------- Southwest Airlines ----------------
    {
        "doc_id": "wn-contract-of-carriage",
        "path": "airline_contracts/southwest/contract_of_carriage_southwest.pdf",
        "doc_title": "Southwest Airlines — Contract of Carriage",
        "publisher": "Southwest Airlines",
        "jurisdiction": "US",
        "airline_iata": "WN",
        "doc_type": "contract",
        "source_url": "https://www.southwest.com/swa-resources/pdfs/corporate-commitments/contract-of-carriage.pdf",
        "effective_date": "2026-09-03",
    },
    {
        "doc_id": "wn-customer-service-plan",
        "path": "airline_contracts/southwest/customer_service_plan_southwest.pdf",
        "doc_title": "Southwest Airlines — Customer Service Plan",
        "publisher": "Southwest Airlines",
        "jurisdiction": "US",
        "airline_iata": "WN",
        "doc_type": "service_plan",
        "source_url": "https://www.southwest.com/swa-resources/pdfs/corporate-commitments/customer-service-plan.pdf",
        "effective_date": "2026-09-03",
    },

    # ---------------- United Airlines ----------------
    {
        "doc_id": "ua-contract-of-carriage",
        "path": "airline_contracts/united/contract_of_carriage _ United.pdf",
        "doc_title": "United Airlines — Contract of Carriage",
        "publisher": "United Airlines",
        "jurisdiction": "US",
        "airline_iata": "UA",
        "doc_type": "contract",
        "source_url": "https://www.united.com/en/us/fly/contract-of-carriage.html",
        "effective_date": "2026-09-01",
    },
    {
        "doc_id": "ua-customer-commitment",
        "path": "airline_contracts/united/customer_commitment_United.md",
        "doc_title": "United Airlines — Customer Commitment",
        "publisher": "United Airlines",
        "jurisdiction": "US",
        "airline_iata": "UA",
        "doc_type": "service_plan",
        "source_url": "https://www.united.com/en/us/fly/customer-commitment.html",
        "effective_date": "2026-09-01",
    },
]

# PDF parsing overrides are document-specific because layouts differ
# "sorted" fixes extraction order for affected eCFR PDFs.
# "after_page_number" removes extracted page footnotes where required.
META_KEYS = (
    "doc_id", "doc_title", "publisher", "jurisdiction",
    "airline_iata", "doc_type", "source_url", "effective_date",
)


def parse_corpus():
    """
    Parse every manifest entry into Sections.

    Returns ({doc_id: sections}, problems). Chunk-size independent: the golden set
    resolves its gold passages here, so one gold target serves every chunk size.
    """
    from flight_delay.ingest import parse_document

    parsed: dict[str, list] = {}
    problems: list[str] = []
    for entry in CORPUS:
        fpath = DATA / entry["path"]
        if not fpath.exists():
            problems.append(f"missing: {entry['path']}")
            print(f"  MISSING  {entry['doc_id']:28} {entry['path']}")
            continue

        meta = {k: entry.get(k, "") for k in META_KEYS}
        meta["effective_date"] = entry.get("effective_date")
        if entry.get("pdf_reading_order"):
            meta["reading_order"] = entry["pdf_reading_order"]
        if entry.get("pdf_footnotes"):
            meta["footnotes"] = entry["pdf_footnotes"]

        try:
            parsed[entry["doc_id"]] = parse_document(fpath, **meta)
        except Exception as e:
            problems.append(f"parse failed: {entry['doc_id']} ({type(e).__name__}: {e})")
            print(f"  ERROR    {entry['doc_id']:28} {type(e).__name__}: {e}")
    return parsed, problems


def build_chunks(settings):
    """
    Parse and chunk the full manifest, returning chunks and validation problems
    """
    from flight_delay.embeddings import chunk_token_counter
    from flight_delay.ingest import chunk_sections

    parsed, problems = parse_corpus()
    _counter_name, count_tokens = chunk_token_counter(settings)

    all_chunks = []
    for entry in CORPUS:
        sections = parsed.get(entry["doc_id"])
        if sections is None:
            continue
        chunks = chunk_sections(
            sections,
            target_tokens=settings.chunk_target_tokens,
            overlap_pct=settings.chunk_overlap_pct,
            breadcrumb=settings.chunk_breadcrumb,
            count_tokens=count_tokens,
        )
        if not chunks:
            problems.append(f"no chunks: {entry['doc_id']} ({len(sections)} sections)")
        kind = "LAW " if not entry.get("airline_iata") else entry["airline_iata"] + "  "
        print(f"  {kind} {entry['doc_id']:28} {len(sections):4} sections -> {len(chunks):5} chunks")
        all_chunks.extend(chunks)

    return all_chunks, problems


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def corpus_digest() -> str:
    """
    Hash corpus files and manifest metadata into a build identity
    """
    parts = [f"{json.dumps(e, sort_keys=True, default=str)}|{_file_sha256(DATA / e['path'])}"
             for e in CORPUS]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def index_identity(settings, embedder, chunks) -> dict:
    """Record the configuration and sources used to build the index"""
    from flight_delay.embeddings import chunk_token_counter_name
    from flight_delay.ingest import PARSER_VERSION

    return {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus_digest": corpus_digest(),
        "source_sha256": {e["doc_id"]: _file_sha256(DATA / e["path"]) for e in CORPUS},
        "parser_version": PARSER_VERSION,
        "chunk_target_tokens": settings.chunk_target_tokens,
        "chunk_overlap_pct": settings.chunk_overlap_pct,
        "chunk_breadcrumb": settings.chunk_breadcrumb,
        "chunk_token_counter": chunk_token_counter_name(settings),
        "embedder": settings.embedder,
        "embedding_model": settings.embedding_model,
        "embedding_revision": getattr(embedder, "revision", None),
        "embedding_dim": embedder.dim,
        "n_chunks": len(chunks),
        "doc_chunks": {e["doc_id"]: sum(c.doc_id == e["doc_id"] for c in chunks) for e in CORPUS},
    }


def report_coverage(chunks) -> list[str]:
    """
    Report corpus coverage and return problems that block activation
    """
    from flight_delay.tools import SUPPORTED_AIRLINES

    gov = [c for c in chunks if c.source_class == "government"]
    air = [c for c in chunks if c.source_class == "airline"]

    print(f"\n  government (LAW) : {len(gov):5} chunks")
    by_juris: dict[str, int] = {}
    for c in gov:
        by_juris[c.jurisdiction] = by_juris.get(c.jurisdiction, 0) + 1
    for j, n in sorted(by_juris.items()):
        print(f"      {j:20} {n:5}")

    print(f"  airline policy   : {len(air):5} chunks")
    by_airline: dict[str, int] = {}
    for c in air:
        by_airline[c.airline_iata] = by_airline.get(c.airline_iata, 0) + 1
    for a, n in sorted(by_airline.items()):
        print(f"      {a:20} {n:5}")

    problems = []
    for region in ("US", "EU", "UK"):
        if not by_juris.get(region):
            problems.append(f"no government chunks for {region}: its passengers' entitlements "
                            "cannot be stated")
    for iata in sorted(SUPPORTED_AIRLINES):
        if not by_airline.get(iata):
            problems.append(f"no chunks for supported airline {iata}")
    for p in problems:
        print(f"  ERROR: {p}")
    return problems


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="accepted for existing commands; every build replaces the corpus "
                         "atomically and never touches conversations")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and report only; no embedding, no database")
    ap.add_argument("--manifest-out", default=str(MANIFEST_OUT),
                    help="where to write the build manifest (never inside data/)")
    args = ap.parse_args(argv)

    from flight_delay.config import get_settings

    settings = get_settings()
    print(f"chunking: target={settings.chunk_target_tokens}tok "
          f"overlap={settings.chunk_overlap_pct}% breadcrumb={settings.chunk_breadcrumb}")

    print("\n--- Parsing corpus ---")
    chunks, problems = build_chunks(settings)
    problems += report_coverage(chunks) if chunks else ["no chunks produced"]

    toks = [c.token_estimate for c in chunks] or [0]
    print(f"\n  total: {len(chunks)} chunks | tokens "
          f"min={min(toks)} median={sorted(toks)[len(toks)//2]} max={max(toks)}")

    if problems:
        print("\nINDEX NOT BUILT - fix these first (the live index was not touched):")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    if args.dry_run:
        print("\nDry run: nothing embedded, nothing written.")
        return

    manifest_out = Path(args.manifest_out).resolve()
    if manifest_out.is_relative_to(DATA.resolve()):
        raise SystemExit(f"--manifest-out must not be inside {DATA}: data/ is protected source material")

    from flight_delay.embeddings import build_embedder
    from flight_delay.store import PgVectorStore

    if settings.embedder == "hash" and not settings.allow_test_doubles:
        raise SystemExit("EMBEDDER=hash builds a test-double index (hash vectors are not semantic). "
                         "Set ALLOW_TEST_DOUBLES=true for a smoke index, or use EMBEDDER=st.")

    print(f"\nembedder={settings.embedder} model={settings.embedding_model}")
    embedder = build_embedder(settings)
    identity = index_identity(settings, embedder, chunks)
    store = PgVectorStore(settings.pg_dsn)

    print(f"Embedding {len(chunks)} chunks (dim={embedder.dim}) ...")
    t0 = time.time()
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    print(f"  embedded in {time.time()-t0:.1f}s")

    t0 = time.time()
    staging = store.begin_staging(embedder.dim)
    try:
        B = 500
        for i in range(0, len(chunks), B):
            staging.add(chunks[i:i+B], vectors[i:i+B])
        total = staging.activate(identity, expected_chunks=len(chunks),
                                 required_docs=[e["doc_id"] for e in CORPUS])
    except Exception:
        staging.abandon()
        print("\nINDEX NOT ACTIVATED: the staged build was discarded; the live index is unchanged.")
        raise
    print(f"  staged, validated and activated in {time.time()-t0:.1f}s")
    print(f"\nINDEX READY: {total} chunks (corpus {identity['corpus_digest']}, "
          f"parser {identity['parser_version']})")

    manifest = {"documents": CORPUS, "identity": identity, "total_chunks": total}
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Build manifest -> {manifest_out}")


if __name__ == "__main__":
    main()
