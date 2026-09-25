"""
parse regulation/policy documents, chunk them for retrieval.

"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .models import Chunk, Section


PARSER_VERSION = "2026-09-19.1"

# --------------------------------------------------------------------------
# Token estimation 
# --------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# PDF parsing (airline contracts of carriage, customer service plans)
# --------------------------------------------------------------------------

# Regulation headings: "§ 259.4 Contingency Plan...", "Article 7 Right to
# compensation". Checked FIRST because it is the most precise signal available —
# when it matches, the document's own numbering is doing the work
_REG_HEADING_RE = re.compile(
    r"^(?:§\s*\d+\.\d+[a-z]?|Article\s+\d+[a-z]?)\s+[A-Z].{0,100}$",
    re.MULTILINE,
)

# Numbered headings: "1. Offering the lowest fare available", "12. Tarmac
# delays". Customer service plans are written as a numbered list of
# commitments, so the number IS the section structure. Allowed to run long
# because these headings are sometimes a full sentence of regulatory reference.
_NUMBERED_HEADING_RE = re.compile(r"^\d{1,2}\.\s+[A-Z].{0,200}$", re.MULTILINE)

# Decimal outline headings: "4.4.7. `Long delays' at arrival" (the number may
# stand alone with the title on the next line) under upper-case chapters
# "4. PASSENGERS' RIGHTS". The Commission's 261/2004 guidelines are numbered this
# way; under the single-level tier its 36 pages became 10 sections of up to 40k
# characters. A single-level number counts only with an all-capitals title, so
# an ordinary numbered list item ("1. The passenger ...") never qualifies.
_DECIMAL_HEADING_RE = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2}){1,3}\.(?:[ \t]+\S.{0,250})?"
    r"|\d{1,2}\.[ \t]+[A-Z][A-Z0-9 ,'’()/—–-]{3,120})$",
    re.MULTILINE,
)

# Contract rules: Delta prints "RULE 19:  FLIGHT DELAYS/CANCELLATIONS" (title
# sometimes on the next line), United "Rule 24 Flight Delays/Cancellations/...".
# Tried BEFORE numbered headings: without it, the numbered list items inside
# each rule ("1. The first flight on which space is available; or") became the
# sections, filing several rules' text under a fragment of one sentence.
# "Rule 12(E) applies ..." is a cross-reference, not a heading: a title must
# follow the number directly and carry no full stop, comma or parenthesis.
_RULE_HEADING_RE = re.compile(
    r"^(?:RULE[ \t]+\d+[A-Z]?[ \t]*:[^\n]{0,100}|Rule[ \t]+\d+[ \t]+[A-Z][^.,()\n]{2,100})$",
    re.MULTILINE,
)

# Heading patterns for airline policy documents. These are broad enough to
# catch "Delays, cancellations and diversions", "Baggage liability", etc.
_PDF_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:Accommodation|Assistance|Baggage|Check-in|Customer|Delays?|Diversion|"
    r"Essential|Family|Flights?|Guaranteed|Handling|Lowest|Other|Oversold|"
    r"Refund|Service|Ticket|Tarmac|Unaccompanied|24-hour)"
    r"[A-Za-z ,/&()'-]*"
    r")$",
    re.MULTILINE,
)


_GENERIC_HEADING_RE = re.compile(
    r"^([A-Z][A-Za-z ,/&()'-]{5,80})$", re.MULTILINE
)


def parse_pdf(
    pdf_path: str | Path,
    *,
    doc_id: str,
    doc_title: str,
    publisher: str,
    jurisdiction: str = "US",
    airline_iata: str = "",
    doc_type: str = "",
    source_url: str = "",
    effective_date: str | None = None,
    reading_order: str = "extraction",
    footnotes: str = "keep",
) -> list[Section]:
    """
    Extract sections from a PDF (contracts, service plans, regulation prints).

    """
    if reading_order not in ("extraction", "sorted"):
        raise ValueError(f"reading_order must be 'extraction' or 'sorted', not {reading_order!r}")
    if footnotes not in ("keep", "after_page_number"):
        raise ValueError(f"footnotes must be 'keep' or 'after_page_number', not {footnotes!r}")
    try:
        import fitz  
    except ImportError as e:
        raise ImportError(
            "pymupdf (PyMuPDF) is required for PDF parsing. "
            "Install with: pip install pymupdf\n"
            "Or save the document as plain text and use sections_from_plaintext()."
        ) from e

    doc = fitz.open(str(pdf_path))
    pages = [page.get_text("text", sort=reading_order == "sorted") for page in doc]
    doc.close()
    # Sorted extraction keeps layout indentation
    pages = ["\n".join(ln.strip() for ln in page.splitlines()) for page in pages]
    if footnotes == "after_page_number":
        pages = [_drop_after_page_number(page) for page in pages]

    # Before combining the pages, remove repeated headers and footers.
    full_text = "\n".join(_strip_page_furniture(pages))

    # Clean up common PDF artifacts
    full_text = strip_legislation_annotations(full_text)
    full_text = _repair_glyphs(full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r"[ \t]+", " ", full_text)

    return _split_into_sections(
        full_text,
        doc_id=doc_id,
        doc_title=doc_title,
        publisher=publisher,
        jurisdiction=jurisdiction,
        airline_iata=airline_iata,
        doc_type=doc_type,
        source_url=source_url,
        effective_date=effective_date,
    )


def _repair_glyphs(text: str) -> str:
    """
    Restore characters that PDF extraction lost to U+FFFD.

    Both `§` and `—` come back as the replacement character from these sources,
    and the difference matters: `§ 259.4` is a heading anchor and a retrieval
    token that users type verbatim, while a mangled dash is only cosmetic. So
    disambiguate by context — a replacement char followed by a section number is
    a section symbol; everything else is a dash..� 259.4 
    Contingency plans becomes § 259.4 Contingency plans but Passengers
     � including those with connections becomes 
     Passengers — including those with connections
    """
    text = re.sub(r"�(?=\s*\d+\.\d+)", "§", text)
    # Between two letters it was an apostrophe ("passenger�s", "don�t"); a dash
    # never sits between letters without spaces.
    text = re.sub(r"(?<=[A-Za-z])�(?=[A-Za-z])", "'", text)
    return text.replace("�", "—")


_PAGE_OF_TOTAL_RE = re.compile(r"^\d{1,4}\s*/\s*\d{1,4}$")


_FOOTNOTE_START_RE = re.compile(r"^\(\d{1,3}\)\s")


def _drop_after_page_number(page: str) -> str:
    """
    Cut a page at its first footnote line after the last "n/N" page number
    (see parse_pdf `footnotes`). Footer lines printed after the page number
    ("ELI: ...") stay, so they repeat on every page and go as furniture.
    """
    lines = page.split("\n")
    idx = [i for i, ln in enumerate(lines) if _PAGE_OF_TOTAL_RE.match(ln.strip())]
    if not idx:
        return page
    cut = next((j for j in range(idx[-1] + 1, len(lines))
                if _FOOTNOTE_START_RE.match(lines[j].strip())), len(lines))
    return "\n".join(lines[:cut])


# legislation.gov.uk prints (UK261) follow each article with a "Textual
# Amendments" block of S.I. citations, and mark amended text inline with the
# block's ids: "[F143 In case of a delay ..." is marker F14 then paragraph 3.
_ANNOTATION_HEAD_RE = re.compile(r"^Textual Amendments$", re.MULTILINE)
_ANNOTATION_END_RE = re.compile(r"^(?:Article\s+\d+|ANNEX)\b")
_ANNOTATION_ID_RE = re.compile(r"^F(\d{1,3})$")
_INLINE_MARKER_RE = re.compile(r"(?<![A-Za-z0-9])\[?F(\d{1,4})(\.\.\.)?")


def strip_legislation_annotations(text: str) -> str:
    """
    Remove legislation.gov.uk amendment annotations; a no-op for other documents.

    """
    if not _ANNOTATION_HEAD_RE.search(text):
        return text
    lines = text.split("\n")
    defined: set[str] = set()
    kept: list[str] = []
    in_block = False
    for ln in lines:
        s = ln.strip()
        if s == "Textual Amendments":
            in_block = True
            continue
        if in_block and _ANNOTATION_END_RE.match(s):
            in_block = False
        if in_block:
            m = _ANNOTATION_ID_RE.match(s)
            if m:
                defined.add(m.group(1))
            continue
        kept.append(ln)

    def unmark(m: re.Match) -> str:
        digits = m.group(1)
        for n in range(len(digits), 0, -1):
            if digits[:n] in defined:
            
                rest = digits[n:]
                return rest + (m.group(2) or "") if rest else ""
        return m.group(0)

    return _INLINE_MARKER_RE.sub(unmark, "\n".join(kept))


# A bare list or paragraph marker: "(a)", "1.", "b)", "2". These repeat on
# nearly every page of a regulation, so repetition alone would call them

_LIST_MARKER_RE = re.compile(r"^(?:\([0-9A-Za-z]{1,4}\)|[0-9A-Za-z]{1,3}[.)]|\d{1,3})$")
_PAGE_NUMBER_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)
# Running headers and footers live in the first/last few lines of a page (the
# Southwest contract's header block is 5 lines deep; the EU Official Journal's
# footer 6 lines up); page numbers in the very first/last lines. Anything
# further in is body text.
_FURNITURE_EDGE_LINES = 6
_PAGE_NUMBER_EDGE_LINES = 2


def _strip_page_furniture(pages: list[str], min_repeats: int = 3) -> list[str]:
    """
    Drop running headers, footers and page numbers.

    """
    if len(pages) < min_repeats:
        return pages

    def edges(lines: list[str], n: int) -> set[int]:
        idx = [i for i, ln in enumerate(lines) if ln.strip()]
        return set(idx[:n]) | set(idx[-n:])

    split = [page.splitlines() for page in pages]
    counts: dict[str, int] = {}
    for lines in split:
        for line in {lines[i].strip() for i in edges(lines, _FURNITURE_EDGE_LINES)}:
            counts[line] = counts.get(line, 0) + 1

    threshold = max(min_repeats, int(len(pages) * 0.6))
    furniture = {
        line for line, n in counts.items()
        if n >= threshold and len(line) <= 120 and not _LIST_MARKER_RE.match(line)
    }

    out = []
    for lines in split:
        edge = edges(lines, _FURNITURE_EDGE_LINES)
        number_edge = edges(lines, _PAGE_NUMBER_EDGE_LINES)
        kept = [
            ln for i, ln in enumerate(lines)
            if not (i in edge and ln.strip() in furniture)
            and not (i in number_edge and _PAGE_NUMBER_RE.match(ln.strip()))
        ]
        out.append("\n".join(kept))
    return out


def _heading_key(heading: str) -> str:
    """The numbering a heading is filed under ("rule 24", "§250.7", "article 7"), else its text."""
    m = re.match(r"^\s*(rule\s+\d+[a-z]?|§\s*\d+\.\d+[a-z]?|article\s+\d+[a-z]?)\b", heading, re.I)
    key = m.group(1) if m else heading
    return re.sub(r"\s+", " ", key).lower().replace("§ ", "§")


def _complete_headings(text: str, matches: list[re.Match]) -> list[tuple[int, int, str]]:
    """
    (start, end, heading) for each match, with wrapped headings completed.

    eCFR section headings always end with a full stop; a PDF line break inside
    one ("§ 250.5 Amount of denied boarding compensation for passengers denied
    boarding" / "involuntarily.") used to leave the rest of the title as the
    first line of the section body. 
    """
    out = []
    for m in matches:
        start, end, heading = m.start(), m.end(), m.group()
        if heading.lstrip().startswith("§") and not heading.rstrip().endswith("."):
            for _ in range(2):
                nl = text.find("\n", end + 1)
                line_end = nl if nl != -1 else len(text)
                line = text[end:line_end].strip()
                if not line or len(line) > 100 or _REG_HEADING_RE.match(line):
                    break
                heading, end = f"{heading} {line}", line_end
                if line.endswith("."):
                    break
        elif re.fullmatch(r"\s*\d{1,2}(?:\.\d{1,2}){1,3}\.\s*", heading):
            nl = text.find("\n", end + 1)
            line_end = nl if nl != -1 else len(text)
            line = text[end:line_end].strip()
            if line and len(line) <= 250:
                heading, end = f"{heading.strip()} {line}", line_end
        elif re.fullmatch(r"\s*RULE\s+\d+[A-Z]?\s*:\s*", heading):
            nl = text.find("\n", end + 1)
            line_end = nl if nl != -1 else len(text)
            line = text[end:line_end].strip()
            if line and line.upper() == line and len(line) <= 100 and not line.startswith("RULE"):
                heading, end = f"{heading.strip()} {line}", line_end
        out.append((start, end, heading))
    return out


def _drop_contents_entries(text: str, heads: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """
    Keep ONE heading per rule/section number: its last real occurrence.
    """
    real: dict[str, int] = {}
    for i, (_s, e, h) in enumerate(heads):
        nxt = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        body = text[e:nxt].strip()
        first_line = body.split("\n", 1)[0]
        if len(body) < 30 or re.search(r"\.{4,}", h) or re.search(r"\.{4,}", first_line):
            continue
        real[_heading_key(h)] = i
    return [heads[i] for i in sorted(real.values())]


def _drop_short_entries(text: str, heads: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Drop headings followed by under 30 characters before the next one"""
    out = []
    for i, (s, e, h) in enumerate(heads):
        nxt = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        if len(strip_contents_lines(text[e:nxt])) >= 30:
            out.append((s, e, h))
    return out


_CONTENTS_LINE_RE = re.compile(r"(?:\.\s?){4,}\s*[0-9ivxlc]{1,4}\s*$", re.IGNORECASE)
_ORPHAN_MARKER_RE = re.compile(r"^\s*(?:[0-9]{1,3}|[a-z]|[ivx]{1,4})[.)]\s*$", re.IGNORECASE)


def strip_contents_lines(text: str) -> str:
    """
    Remove table-of-contents lines ("Check-in ........ 19") from a section body.

    """
    lines = text.split("\n")
    drop = [bool(_CONTENTS_LINE_RE.search(ln)) for ln in lines]
    for i, ln in enumerate(lines):
        if not drop[i] and _ORPHAN_MARKER_RE.match(ln):
            nxt = next((j for j in range(i + 1, len(lines)) if lines[j].strip()), None)
            if nxt is not None and drop[nxt]:
                drop[i] = True
    return "\n".join(ln for ln, d in zip(lines, drop, strict=True) if not d).strip()


def _split_into_sections(
    text: str,
    *,
    doc_id: str,
    doc_title: str,
    publisher: str,
    jurisdiction: str,
    airline_iata: str,
    source_url: str,
    effective_date: str | None,
    doc_type: str = "",
) -> list[Section]:
    """
    Split extracted text into sections by detecting heading lines.

    Tiers, strongest first; the first that finds at least 3 headings wins:
    regulation numbering (§ / Article), contract rules (Rule N), numbered
    commitments, airline policy keywords, generic title-case lines. If none
    does, the whole document is one section.
    """
    heads: list[tuple[int, int, str]] = []
    for pattern in (_REG_HEADING_RE, _RULE_HEADING_RE, _DECIMAL_HEADING_RE,
                    _NUMBERED_HEADING_RE, _PDF_HEADING_RE, _GENERIC_HEADING_RE):
        found = _complete_headings(text, list(pattern.finditer(text)))
        if pattern in (_REG_HEADING_RE, _RULE_HEADING_RE):
            found = _drop_contents_entries(text, found)
        else:
            # "1. Introduction ........ 4" (or ". . . . 4") is a contents line,
            # whatever the tier.
            found = [h for h in found if not re.search(r"(?:\.\s?){4,}", h[2])]
        if pattern is _DECIMAL_HEADING_RE:
            # Only a document that is really outlined: three sub-level headings.
            found = _drop_short_entries(text, found)
            # A wrapped contents entry keeps its leader on the next line, so it
            # survives both filters; the contents come first, so keep the LAST
            # heading per outline number.
            last = {re.match(r"\s*([\d.]+)", h[2]).group(1): i for i, h in enumerate(found)}
            found = [found[i] for i in sorted(last.values())]
            if sum(1 for h in found if re.match(r"\s*\d{1,2}\.\d", h[2])) < 3:
                continue
        if len(found) >= 3 or pattern is _GENERIC_HEADING_RE:
            heads = found
            break

    if not heads:
        # Whole document as one section
        return [Section(
            doc_id=doc_id, doc_title=doc_title, publisher=publisher,
            jurisdiction=jurisdiction, airline_iata=airline_iata,
            doc_type=doc_type,
            section_id="full-document", heading=doc_title,
            path=(doc_title,), text=text.strip(),
            source_url=source_url, effective_date=effective_date,
        )]

    sections: list[Section] = []

    # Text before the first heading
    preamble = strip_contents_lines(text[:heads[0][0]])
    if preamble and len(preamble) >= 30:
        sections.append(Section(
            doc_id=doc_id, doc_title=doc_title, publisher=publisher,
            jurisdiction=jurisdiction, airline_iata=airline_iata,
            doc_type=doc_type,
            section_id="preamble", heading="Preamble",
            path=(doc_title, "Preamble"), text=preamble,
            source_url=source_url, effective_date=effective_date,
        ))

    for i, (_start, end, raw) in enumerate(heads):
        # A heading may wrap across lines in the PDF 
        heading = re.sub(r"\s+", " ", raw).strip()
        stop = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        body = strip_contents_lines(text[end:stop])

        if not body or len(body) < 30:
            continue

        sec_id = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")[:60]

        sections.append(Section(
            doc_id=doc_id, doc_title=doc_title, publisher=publisher,
            jurisdiction=jurisdiction, airline_iata=airline_iata,
            doc_type=doc_type,
            section_id=sec_id, heading=heading,
            path=(doc_title, heading), text=body,
            source_url=source_url, effective_date=effective_date,
        ))

    return sections


# --------------------------------------------------------------------------
# Markdown parsing
# --------------------------------------------------------------------------

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def sections_from_markdown(
    text: str,
    *,
    doc_id: str,
    doc_title: str,
    publisher: str,
    jurisdiction: str = "US",
    airline_iata: str = "",
    doc_type: str = "",
    source_url: str = "",
    effective_date: str | None = None,
) -> list[Section]:
    """
    Split Markdown on ATX headings, keeping the heading hierarchy in `path`.
    """
    lines = text.splitlines()
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []       # (level, heading)
    cur_heading = "Preamble"
    body: list[str] = []
    in_code_fence = False

    def flush():
        content = "\n".join(body).strip()
        if not content:
            return
        trail = [h for _, h in stack]
        path = (doc_title, *trail) if trail else (doc_title, cur_heading)
        sec_id = re.sub(r"[^a-z0-9]+", "-", cur_heading.lower()).strip("-")[:60]
        sections.append(Section(
            doc_id=doc_id, doc_title=doc_title, publisher=publisher,
            jurisdiction=jurisdiction, airline_iata=airline_iata,
            doc_type=doc_type,
            section_id=sec_id or "preamble", heading=cur_heading,
            path=path, text=content,
            source_url=source_url, effective_date=effective_date,
        ))

    for line in lines:
        # A '#' inside a fenced code block is not a heading.
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
            body.append(line)
            continue

        m = None if in_code_fence else _ATX_RE.match(line)
        if not m:
            body.append(line)
            continue

        flush()
        level, heading = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        cur_heading, body = heading, []

    flush()
    return sections


# --------------------------------------------------------------------------
# eCFR XML parsing (government regulations)
# --------------------------------------------------------------------------

def fetch_ecfr_part(title: int, part: int, date: str, timeout: float = 60.0) -> bytes:
    import httpx
    url = f"https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
    resp = httpx.get(
        url, params={"part": str(part)}, timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "flight-delay-rag/0.1 (educational project)"},
    )
    resp.raise_for_status()
    return resp.content


def parse_ecfr_xml(
    xml_bytes: bytes,
    *,
    doc_id: str,
    doc_title: str,
    publisher: str = "US DOT",
    jurisdiction: str = "US",
    airline_iata: str = "",
    doc_type: str = "regulation",
    source_url: str = "",
    effective_date: str | None = None,
) -> list[Section]:
    from lxml import etree

    def _clean(text):
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _element_text(el):
        return _clean(" ".join(el.itertext()))

    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(xml_bytes, parser=parser)
    sections: list[Section] = []

    part_el = root.find(".//DIV5")
    part_label = ""
    if part_el is not None:
        part_n = part_el.get("N", "")
        part_head = _clean(part_el.findtext("HEAD"))
        part_label = part_head or (f"Part {part_n}" if part_n else "")

    for sec_el in root.iter("DIV8"):
        sec_n = sec_el.get("N", "").strip()
        heading = _clean(sec_el.findtext("HEAD"))

        subpart_label = ""
        parent = sec_el.getparent()
        while parent is not None:
            if parent.tag == "DIV6":
                subpart_label = _clean(parent.findtext("HEAD"))
                break
            parent = parent.getparent()

        paragraphs = []
        for p in sec_el.iter("P"):
            txt = _element_text(p)
            if txt and not txt.startswith("[") and len(txt) > 3:
                paragraphs.append(txt)

        if not paragraphs:
            continue

        body = "\n\n".join(paragraphs)

        if heading.lstrip().startswith("§") or heading.startswith(sec_n):
            sec_label = heading
        else:
            sec_label = f"§ {sec_n} {heading}".strip()

        path = tuple(x for x in (part_label, subpart_label, sec_label) if x)

        sections.append(Section(
            doc_id=doc_id, doc_title=doc_title, publisher=publisher,
            jurisdiction=jurisdiction, airline_iata=airline_iata, doc_type=doc_type,
            section_id=sec_n or heading[:40], heading=heading,
            path=path, text=body, source_url=source_url,
            effective_date=effective_date,
        ))

    return sections


# --------------------------------------------------------------------------
# Plain text fallback
# --------------------------------------------------------------------------

# "Rule 24", "Article 7", "§ 250.5", "PART 250—OVERSALES"
_LABELLED_HEADING_RE = re.compile(
    r"^\s*((?:Rule|Article|Section|Chapter|Part|§)\s*[0-9IVXA-Z]+[.:—–-]?.*)$",
    re.IGNORECASE,
)
# A line in ALL CAPS on its own: how scraped help-pages and the EU/AirHelp-style
# exports mark their sections. Requires a letter so "2024" is not a heading.
_CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 ,'’&/()\-—–.]{4,79}$")
# Setext style: a line underlined by --- or === on the following line.
_SETEXT_UNDERLINE_RE = re.compile(r"^\s*([=\-]){3,}\s*$")


def sections_from_plaintext(
    text: str,
    *,
    doc_id: str,
    doc_title: str,
    publisher: str,
    jurisdiction: str = "US",
    airline_iata: str = "",
    doc_type: str = "",
    source_url: str = "",
    effective_date: str | None = None,
) -> list[Section]:
    """
    Heading detection for unstructured text
    """
    lines = text.splitlines()
    heading_idx = _detect_plaintext_headings(lines)

    sections: list[Section] = []
    cur_head, cur_body = "Preamble", []

    def flush():
        body = "\n\n".join(b for b in cur_body if b.strip())
        if body.strip():
            sec_id = re.sub(r"[^a-z0-9]+", "-", cur_head.lower()).strip("-")[:60]
            sections.append(Section(
                doc_id=doc_id, doc_title=doc_title, publisher=publisher,
                jurisdiction=jurisdiction, airline_iata=airline_iata,
                doc_type=doc_type,
                section_id=sec_id or "preamble", heading=cur_head,
                path=(doc_title, cur_head), text=body,
                source_url=source_url, effective_date=effective_date,
            ))

    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        stripped = line.strip()
        if i in heading_idx:
            flush()
            cur_head = _clean_heading(stripped)
            cur_body = []
            # Swallow a ==== / ---- underline belonging to a setext heading.
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            skip_next = bool(_SETEXT_UNDERLINE_RE.match(nxt))
        else:
            cur_body.append(stripped)
    flush()
    return sections


def _clean_heading(line: str) -> str:
    """Strip the markup off a heading line so breadcrumbs read cleanly."""
    line = _ATX_RE.sub(r"\2", line.strip())
    return line.strip().strip("*_").strip()


def _detect_plaintext_headings(lines: list[str]) -> set[int]:
    """
    Find heading line numbers, strongest signal first.

    """
    atx: set[int] = set()
    strong: set[int] = set()
    weak: set[int] = set()

    def next_nonblank(start: int) -> str:
        for j in range(start, min(start + 5, len(lines))):
            if lines[j].strip():
                return lines[j].strip()
        return ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        body_ahead = next_nonblank(i + 1)

        if _ATX_RE.match(stripped):
            atx.add(i)
            continue

        if len(stripped) < 120 and (
            _SETEXT_UNDERLINE_RE.match(nxt)
            or _LABELLED_HEADING_RE.match(line)
            or (_CAPS_HEADING_RE.match(stripped) and any(c.isalpha() for c in stripped))
        ):
            strong.add(i)
            continue

      
        if (
            10 <= len(stripped) <= 90
            and stripped[-1] not in ".:;,!?"
            and not stripped[0].isdigit()
            and stripped[0].isupper()
            and len(body_ahead) > 120
        ):
            weak.add(i)

    if len(atx) >= 2:
        return atx
    if len(strong) >= 3:
        return strong | atx
    if len(strong | atx) >= 2:
        return strong | atx
    return weak | strong | atx


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

def parse_document(path: str | Path, **meta) -> list[Section]:
    """
    Parse any supported document into Sections, choosing the parser by suffix.

    """
    return unique_section_ids(_parse_document(Path(path), **meta))


def _parse_document(path: Path, **meta) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix != ".pdf":
        meta.pop("reading_order", None)
        meta.pop("footnotes", None)

    if suffix == ".pdf":
        return parse_pdf(path, **meta)
    if suffix == ".xml":
        return parse_ecfr_xml(path.read_bytes(), **meta)

    text = path.read_text(encoding="utf-8", errors="replace")

    # Trust content over extension. Scraped ".md" files often carry no headings
    # at all, and hand-cleaned ".txt" files often carry Markdown ones — so pick
    # the parser that actually finds structure rather than the one the suffix
    # implies. A single-section result means the parser found nothing.
    if re.search(r"^#{1,6}\s+\S", text, re.MULTILINE):
        sections = sections_from_markdown(text, **meta)
        if len(sections) > 1:
            return sections
    return sections_from_plaintext(text, **meta)


def unique_section_ids(sections: list[Section]) -> list[Section]:
    """
    Make section_id unique within each document: a repeat becomes "<id>-2", "-3"...
    """
    seen: dict[tuple[str, str], int] = {}
    out = []
    for s in sections:
        key = (s.doc_id, s.section_id)
        seen[key] = seen.get(key, 0) + 1
        out.append(s if seen[key] == 1 else replace(s, section_id=f"{s.section_id}-{seen[key]}"))
    return out


def with_effective_date(sections: list[Section], date: str) -> list[Section]:
    return [replace(s, effective_date=date) for s in sections]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

TokenCounter = Callable[[str], int]

# Where a paragraph block may be split without cutting a sentence, strongest
# first: a line that starts a list item ("(a)", "(2)", "b.", "3)"), then the
# end of a sentence or clause. 
_LIST_ITEM_SPLIT_RE = re.compile(r"\n(?=\(?(?:[0-9]{1,3}|[a-zA-Z]{1,2}|[ivx]{1,5})[.)]\s)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+")


def _fit_units(text: str, max_tokens: int, count: TokenCounter,
               splitters: tuple[re.Pattern, ...]) -> list[str]:
    """Split `text` at the strongest boundary that makes the pieces fit; recurse."""
    if count(text) <= max_tokens or not splitters:
        return [text]
    head, rest = splitters[0], splitters[1:]
    parts = [p.strip() for p in head.split(text) if p.strip()]
    if len(parts) == 1:
        return _fit_units(text, max_tokens, count, rest)
    out: list[str] = []
    for part in parts:
        out.extend(_fit_units(part, max_tokens, count, rest))
    return out


def _split_units(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
 
    units: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block:
            units.extend(_fit_units(block, max_tokens, count,
                                    (_LIST_ITEM_SPLIT_RE, _SENTENCE_SPLIT_RE)))
    return units


def clause_units(text: str) -> list[str]:
    """
    Every paragraph, list item and sentence of `text`, in order: the smallest
    units the chunker ever keeps together. Independent of any chunk size, so the
    golden set anchors its gold passages on them.
    """
    return _split_units(text, 0, len)


def embed_header(section: Section) -> str:
    """
    "[doc title · heading path · effective date]", prepended to the embedded text.

    """
    crumbs = " > ".join(p for p in section.path if p != section.doc_title)
    bits = [section.doc_title, crumbs]
    if section.effective_date:
        bits.append(f"effective {section.effective_date}")
    return "[" + " · ".join(b for b in bits if b) + "]"


def chunk_section(
    section: Section,
    *,
    target_tokens: int = 512,
    overlap_pct: int = 15,
    breadcrumb: bool = True,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """
    Turn one Section into one or more Chunks.

    Invariants:
      1. A chunk NEVER spans two sections. This makes citations correct.
      2. Chunks accumulate whole units (paragraphs, list items, sentences).

    """
    count = count_tokens or estimate_tokens
    units = _split_units(section.text, target_tokens, count)
    overlap_budget = target_tokens * overlap_pct // 100

    def size(parts: list[str]) -> int:
        return count("\n\n".join(parts)) if parts else 0

    groups: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        if current and size([*current, unit]) > target_tokens:
            groups.append(current)
            carried: list[str] = []
            if overlap_budget > 0:
                for prev in reversed(current):
                    if size([prev, *carried]) > overlap_budget:
                        break
                    carried.insert(0, prev)
            if len(carried) == len(current):
                carried = carried[1:]
            while carried and size([*carried, unit]) > target_tokens:
                carried.pop(0)
            current = carried
        current.append(unit)
    if current:
        groups.append(current)

    header = embed_header(section) if breadcrumb else ""
    chunks: list[Chunk] = []
    for i, group in enumerate(groups):
        body = "\n\n".join(group)
        embed_text = f"{header}\n\n{body}" if breadcrumb else body

        digest = hashlib.sha256(
            f"{section.doc_id}|{section.section_id}|{i}|{body}".encode()
        ).hexdigest()[:16]

        chunks.append(Chunk(
            chunk_id=f"{section.doc_id}:{section.section_id}:{i}:{digest}",
            doc_id=section.doc_id,
            doc_title=section.doc_title,
            publisher=section.publisher,
            jurisdiction=section.jurisdiction,
            airline_iata=section.airline_iata,
            doc_type=section.doc_type,
            section_id=section.section_id,
            breadcrumb=section.breadcrumb,
            text=body,
            embed_text=embed_text,
            source_url=section.source_url,
            effective_date=section.effective_date,
            token_estimate=count(body),
        ))

    return chunks


def chunk_sections(sections: list[Section], **kwargs) -> list[Chunk]:
    out: list[Chunk] = []
    for s in sections:
        out.extend(chunk_section(s, **kwargs))
    return out
