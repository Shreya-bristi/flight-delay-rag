#!/usr/bin/env python3
"""
KV-cache saturation benchmark.


USAGE
    # against vLLM
    python scripts/bench_kv_cache.py --base-url http://localhost:8001/v1 \
        --model Qwen/Qwen3-0.6B --metrics-url http://localhost:8001/metrics

    # verify the script itself with no GPU and no server
    python scripts/bench_kv_cache.py --mock

OUTPUT
    data/kv_bench.csv   raw numbers
    data/kv_bench.png   the three-axis plot 
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

# A prompt so each request occupies a realistic amount of KV cache.

PROMPT = (
    "You are reviewing aviation regulations. Summarise, in detail, the "
    "obligations a carrier has to a passenger who has been denied boarding "
    "involuntarily from an oversold flight, including compensation amounts, "
    "the timing of payment, the written notice requirement, and the "
    "circumstances under which compensation is not owed. Be thorough and "
    "explain the reasoning behind each obligation step by step. "
)


# ==========================================================================
# Arithmetic prediction
# ==========================================================================


def predict_knee(
    *,
    vram_gb: float,
    weight_gb: float,
    layers: int,
    kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
    avg_seq_len: int = 1024,
    gpu_util: float = 0.90,
) -> dict:
    """
    Predict the saturation point from first principles.

    The KV cache stores, for every token, a Key and a Value vector per layer per
    KV head:

        bytes_per_token = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes

    Then:
        cache_bytes    = (vram * gpu_util) - weights
        max_tokens     = cache_bytes / bytes_per_token
        knee_requests  = max_tokens / avg_seq_len
    """
    bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    cache_gb = vram_gb * gpu_util - weight_gb
    if cache_gb <= 0:
        return {"error": "weights exceed the memory budget; use quantisation"}
    max_tokens = cache_gb * (1024**3) / bytes_per_token
    return {
        "bytes_per_token": bytes_per_token,
        "kb_per_token": round(bytes_per_token / 1024, 2),
        "cache_gb": round(cache_gb, 2),
        "max_cached_tokens": int(max_tokens),
        "avg_seq_len": avg_seq_len,
        "predicted_knee_concurrency": max(1, int(max_tokens / avg_seq_len)),
    }


# Known model geometries
MODEL_GEOMETRY = {
    "qwen3-0.6b": {"layers": 28, "kv_heads": 8, "head_dim": 128, "weight_gb": 1.2},
    "qwen3-1.7b": {"layers": 28, "kv_heads": 8, "head_dim": 128, "weight_gb": 3.4},
    "qwen3-4b":   {"layers": 36, "kv_heads": 8, "head_dim": 128, "weight_gb": 8.0},
    "qwen3-8b":   {"layers": 36, "kv_heads": 8, "head_dim": 128, "weight_gb": 16.0},
}


# ==========================================================================
# Metrics scraping
# ==========================================================================

INTERESTING = (
    "vllm:gpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:gpu_prefix_cache_hit_rate",
)


def scrape(metrics_url: str) -> dict[str, float]:
    """
    About prometheus
    """
    out: dict[str, float] = {}
    try:
        text = httpx.get(metrics_url, timeout=5.0).text
    except Exception:
        return out
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for name in INTERESTING:
            if line.startswith(name):
                try:
                    out[name] = float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
    return out


# ==========================================================================
# Load generation
# ==========================================================================


def one_request(base_url: str, model: str, max_tokens: int, timeout: float) -> dict:
    """
    Fire one streaming request and measure it with TTFT
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    err = None
    try:
        with httpx.Client(timeout=timeout) as c:
            with c.stream("POST", f"{base_url}/chat/completions", json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        delta = json.loads(body)["choices"][0]["delta"].get("content")
                    except Exception:
                        continue
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        n_tokens += 1
    except Exception as e:
        err = str(e)[:80]
    total = time.perf_counter() - t0
    return {"ttft": ttft, "total": total, "tokens": n_tokens, "error": err}


#(To be continued if needed)