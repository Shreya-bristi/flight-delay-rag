"""
generation.drop_uncited_list_items: the last-resort salvage after every validation
attempt failed (Session 37). Only whole bullet points go, and only when the rest
validates on its own, so no uncited claim can reach the passenger this way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flight_delay.generation import (  # noqa: E402
    build_context,
    drop_uncited_list_items,
    validate_answer,
)
from flight_delay.models import Chunk  # noqa: E402


def _ctx(n: int = 8):
    return build_context([
        Chunk(chunk_id=f"c{i}", doc_id="eu-261", doc_title="Regulation (EC) No 261/2004",
              publisher="EU", jurisdiction="EU", section_id=f"art-{i}", breadcrumb=f"Art {i}",
              text=f"Source text {i}.", embed_text="x", source_url="u", airline_iata="",
              doc_type="regulation", effective_date="2005-02-17", token_estimate=20)
        for i in range(1, n + 1)], None)


# The real gpt-oss-20b answer to the live UA169 VCE->EWR cancellation (Session 37),
# withheld in production for its one uncited, misleading advice bullet.
VENICE = (
    "Your flight departed from Venice (VCE), so EU Regulation 261/2004 applies to you [S2].  \n"
    "By law you are entitled to a full refund of the ticket price and any ancillary fees, and "
    "you may choose either a refund or re-routing to your final destination at the earliest "
    "opportunity [S3].  \n"
    "Because the flight was cancelled, you also qualify for a cash compensation of €600, unless "
    "the airline can prove the cancellation was due to extraordinary circumstances or the "
    "re-routing was within the time limits that allow a 50 % reduction [S7].  \n\n"
    "- Ask United for a refund or re-routing confirmation.  \n"
    "- If you prefer a cash payout, request the €600 compensation.  \n"
    "- Keep any receipts or documentation of the cancellation notice.  \n\n"
    "Do you know whether United offered you a re-routing option?"
)


def test_venice_answer_is_shown_without_its_uncited_bullet():
    ctx = _ctx()
    assert not validate_answer(VENICE, ctx).ok
    text, report, dropped = drop_uncited_list_items(VENICE, ctx)
    assert report.ok and not report.uncited_factual_sentences
    assert dropped == ["- If you prefer a cash payout, request the €600 compensation."]
    assert "cash payout" not in text
    assert "- Ask United for a refund or re-routing confirmation." in text
    assert "[S7]" in text and text.endswith("re-routing option?")


def test_uncited_prose_sentence_is_never_salvaged():
    # Deleting a prose sentence could change what its neighbours mean: withhold instead.
    ctx = _ctx()
    answer = "EU261 applies [S2]. You are owed EUR 600.\n\n- Ask United for the claim form."
    assert drop_uncited_list_items(answer, ctx) is None


def test_invented_marker_is_never_salvaged():
    ctx = _ctx(3)
    answer = "EU261 applies [S9].\n\n- You are owed EUR 600."
    assert drop_uncited_list_items(answer, ctx) is None


def test_nothing_cited_left_is_not_an_answer():
    ctx = _ctx()
    answer = "- You are owed EUR 600.\n- Request the EUR 600 compensation."
    assert drop_uncited_list_items(answer, ctx) is None


def test_a_valid_answer_needs_no_salvage():
    ctx = _ctx()
    assert drop_uncited_list_items("EU261 applies [S2].\n\n- Ask United for the form.", ctx) is None
