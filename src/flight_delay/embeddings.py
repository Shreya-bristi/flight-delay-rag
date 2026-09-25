"""
Embedding backend
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9§$.]+")


def tokenize(text: str) -> list[str]:
    """
    Shared tokeniser for the hash embedder and BM25
    """
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic, offline, no dependencies beyond nump."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.revision = "hash-v1"   

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for tok in tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, n in counts.items():
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            v[idx] += sign * (1.0 + math.log(n))
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def _model_revision(model) -> str | None:
    try:
        config = model[0].auto_model.config
        return getattr(config, "_commit_hash", None)
    except Exception:
        return None


class SentenceTransformerEmbedder:
    """
    The real embedder
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        query_instruction: str = "Represent this sentence for searching relevant passages: ",
        device: str | None = None,
        batch_size: int = 16,
        query_device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer  

        self.model = SentenceTransformer(model_name, device=device)
        self.doc_device = str(self.model.device)
        self.query_device = query_device or self.doc_device
        self.query_instruction = query_instruction
        # Batch 16 at ~512 tokens keeps peak VRAM under ~1.5GB on bge-large,
        # which leaves room for the reranker on a 4GB card
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()
        self.revision = _model_revision(self.model)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._move(self.doc_device)
        # normalize_embeddings=True gives unit vectors, which makes cosine
        # similarity equal to the dot product
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        ).astype(np.float32)

    def _move(self, device: str) -> None:
        if str(self.model.device) != device:
            self.model.to(device)
            if device == "cpu":
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def embed_query(self, text: str) -> np.ndarray:
        self._move(self.query_device)
        return self.model.encode(
            self.query_instruction + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)


def chunk_token_counter_name(settings) -> str:
    """Chunk size units are recorded in the index identity without loading the model"""
    return settings.embedding_model if settings.embedder == "st" else "chars/4"


def chunk_token_counter(settings):
    """
    (name, count) for measuring chunk sizes.
    """
    from .ingest import estimate_tokens

    if settings.embedder != "st":
        return "chars/4", estimate_tokens
    from transformers import AutoTokenizer  

    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])

    return chunk_token_counter_name(settings), count


def build_embedder(settings) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder(dim=settings.embedding_dim)
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        query_instruction=settings.query_instruction,
        query_device=settings.embedding_query_device or None,
    )
