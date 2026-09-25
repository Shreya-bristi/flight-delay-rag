"""
 Confidence gating: decision whether retrieval evidence is strong enough to answer.

The gate runs before generation and abstains when retrieval quality is too weak.
Answer validation runs separately after generation.

Signals:
- no retrieval results
- low reranker score
- weak score separation
- low lexical grounding

Thresholds are configurable and should be calibrated with evaluation data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Chunk

# ignore common words when measuring lexical overlap
_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "should", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your", "am", "about"]
)

_WORD_RE = re.compile(r"[a-z0-9$§.]+")


def _content_words(text: str) -> set[str]:
    return {
        w for w in _WORD_RE.findall(text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }


@dataclass
class ConfidenceDecision:
    should_answer: bool
    confidence: float          # coarse display score, not a probability
    reasons: list[str]         # which signals fired
    signals: dict[str, float]  # raw values, for calibration and Grafana

    @property
    def label(self) -> str:
        if self.should_answer:
            return "high" if self.confidence >= 0.65 else "medium"
        return "low"


class ConfidenceGate:
    def __init__(
        self,
        *,
        min_rerank_score: float = -2.0,
        min_score_margin: float = 0.5,
        min_grounding_ratio: float = 0.34,
        enabled: bool = True,
    ):
        # permissive starting threshold for bge-reranker raw logits
        self.min_rerank_score = min_rerank_score
        self.min_score_margin = min_score_margin
        # require roughly one-third of query content words to be grounded
        self.min_grounding_ratio = min_grounding_ratio
        self.enabled = enabled

    def evaluate(self, question: str, chunks: list[Chunk]) -> ConfidenceDecision:
        signals: dict[str, float] = {}
        reasons: list[str] = []

        if not self.enabled:
            return ConfidenceDecision(True, 1.0, ["gate disabled"], {})

        # -- signal 1: retrieval returned evidence ---------------------------------
        signals["n_chunks"] = float(len(chunks))
        if not chunks:
            return ConfidenceDecision(
                False, 0.0, ["retrieval returned no chunks"], signals
            )

        # -- signal 2: strongest reranker scor --------------------------
      
        scored = [c.rerank_score for c in chunks if c.rerank_score is not None]
        top = max(scored) if scored else None
        if top is not None:
            signals["rerank_top"] = float(top)
            if top < self.min_rerank_score:
                reasons.append(
                    f"top rerank score {top:.2f} below threshold {self.min_rerank_score}"
                )
        else:
            # Continue with the remaining signals if no reranker is configured
            signals["rerank_top"] = float("nan")

        # -- signal 3: separation between reranker score -------------------------------------
        if len(scored) >= 2:
            margin = max(scored) - min(scored)
            signals["score_margin"] = float(margin)
            if margin < self.min_score_margin:
                reasons.append(
                    f"reranker could not discriminate (margin {margin:.2f}); "
                    "question may be out of domain"
                )

        # -- signal 4: lexical grounding --------------------------------
        
        qwords = _content_words(question)
        if qwords:
            pooled: set[str] = set()
            for c in chunks[:3]:
                pooled |= _content_words(c.text)
            grounded = qwords & pooled
            ratio = len(grounded) / len(qwords)
            signals["grounding_ratio"] = round(ratio, 3)
            signals["grounded_words"] = float(len(grounded))
            if ratio < self.min_grounding_ratio:
                reasons.append(
                    f"only {len(grounded)}/{len(qwords)} question terms "
                    f"({ratio:.0%}) appear in retrieved text; below "
                    f"{self.min_grounding_ratio:.0%} threshold"
                )

        # Coarse display score only; not a calibrated probability
        conf = 1.0
        if top is not None:
            conf = min(1.0, max(0.0, (top + 6.0) / 12.0))
        conf *= 0.5 ** len(reasons)

        return ConfidenceDecision(
            should_answer=not reasons,
            confidence=round(conf, 3),
            reasons=reasons,
            signals=signals,
        )


ABSTENTION_TEMPLATE = """I don't have source material that answers this question.

{detail}

What I can suggest:
- Check the airline's Contract of Carriage directly (US carriers are required to publish it under 14 CFR 259.6)
- Call the airline's customer service line and ask them to cite the specific rule
- For a disputed claim, the US DOT complaint process or a consumer-rights lawyer

This system only answers from the aviation regulations it has indexed, and it will not guess at rules it cannot cite."""


def abstention_answer(decision: ConfidenceDecision) -> str:
    """
    Build an abstention response with a reason and next steps
    """
    if decision.reasons and "no chunks" in decision.reasons[0]:
        detail = "Nothing in the indexed regulations matched this question."
    else:
        detail = (
            "The regulations I retrieved were not a close enough match to answer "
            "reliably, and I'd rather say so than guess at a rule."
        )
    return ABSTENTION_TEMPLATE.format(detail=detail)
