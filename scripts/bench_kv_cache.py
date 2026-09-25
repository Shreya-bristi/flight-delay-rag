#!/usr/bin/env python3
"""
KV-cache saturation benchmark.

WHAT THIS PRODUCES
------------------
A concurrency sweep against an OpenAI-compatible LLM server, plotting:

    x = concurrent requests
    y1 = KV cache utilisation      (scraped from vLLM /metrics)
    y2 = p95 time-to-first-token   (measured client-side)
    y3 = output throughput         (tokens/sec, measured client-side)

You will see one shape, and it explains everything about LLM serving cost:
throughput climbs while the cache has room; at ~90-100% cache occupancy the
scheduler starts PREEMPTING requests, the waiting queue deepens, and TTFT
hockey-sticks while throughput flattens. That inflection is the "knee".

WHY THIS IS THE MOST VALUABLE ARTIFACT IN THE PROJECT
-----------------------------------------------------
Anyone can screenshot a latency graph. Almost nobody can predict where the knee
will be BEFORE measuring it, and then show the prediction matching. This script
does the arithmetic first (see predict_knee) and prints both, so your write-up
says "I predicted saturation at 6 concurrent requests and measured 7" instead of
"here is a graph that goes up".

WHY A 4 GB GPU IS AN ADVANTAGE HERE
------------------------------------
KV cache size is (VRAM - weights). On a 4GB card running Qwen3-0.6B in fp16,
weights take ~1.2GB and ~2.4GB remains for cache - so saturation arrives at
4-8 concurrent requests instead of 40-60. The sweep finishes in minutes instead
of an hour, preemption is trivially easy to trigger and observe, and the lesson
is identical. Small hardware makes this experiment cheaper AND clearer.

USAGE
    # against vLLM
    python scripts/bench_kv_cache.py --base-url http://localhost:8001/v1 \
        --model Qwen/Qwen3-0.6B --metrics-url http://localhost:8001/metrics

    # verify the script itself with no GPU and no server
    python scripts/bench_kv_cache.py --mock

OUTPUT
    data/kv_bench.csv   raw numbers
    data/kv_bench.png   the three-axis plot (if matplotlib is installed)
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

# A long-ish prompt so each request occupies a realistic amount of KV cache.
# A two-token prompt would never saturate anything and the sweep would be
# meaningless.
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

    Note kv_heads, not attention heads: modern models use grouped-query
    attention, where several query heads share one KV head. Using the attention
    head count instead would overestimate cache size several-fold - this is the
    most common error in this calculation.

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


# Known model geometries, so you do not have to look them up mid-experiment.
# Verify against the model's config.json - these are for convenience, and a
# wrong number here silently invalidates the prediction.
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
    Pull the gauges we care about out of Prometheus text format.

    Deliberately a tiny hand-rolled parser rather than a dependency: we need
    five gauges, and prometheus_client's parser would be another install for no
    benefit.
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
    Fire one streaming request and measure it.

    TTFT is measured as time-to-first-CONTENT-token, not time to first byte.
    Servers send role/empty deltas first, and counting those would understate
    TTFT by tens of milliseconds - which matters when the whole number is
    sub-second.
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


def mock_request(concurrency: int) -> dict:
    """
    Simulated request, used by --mock to verify the script's own logic.

    The model is deliberately crude but has the RIGHT SHAPE: TTFT is flat while
    there is cache headroom, then grows superlinearly past saturation. This
    exists so the CSV/plot/prediction code can be tested without a GPU.
    """
    knee = 6
    base = 0.18
    if concurrency <= knee:
        ttft = base * (1 + 0.06 * concurrency)
    else:
        ttft = base * (1 + 0.06 * knee) * (1.55 ** (concurrency - knee))
    ttft *= random.uniform(0.92, 1.08)
    time.sleep(0.004)
    return {"ttft": ttft, "total": ttft + 0.9, "tokens": 120, "error": None}


def sweep_level(args, concurrency: int) -> dict:
    results: list[dict] = []
    lock = threading.Lock()
    peak: dict[str, float] = {}
    stop = threading.Event()

    def poller():
        """Sample /metrics DURING the burst. Scraping only afterwards would miss
        the peak entirely, because the cache drains the instant requests end."""
        while not stop.is_set():
            m = scrape(args.metrics_url) if args.metrics_url else {}
            with lock:
                for k, v in m.items():
                    peak[k] = max(peak.get(k, 0.0), v)
            time.sleep(0.25)

    def worker():
        r = mock_request(concurrency) if args.mock else one_request(
            args.base_url, args.model, args.max_tokens, args.timeout
        )
        with lock:
            results.append(r)

    if args.metrics_url and not args.mock:
        threading.Thread(target=poller, daemon=True).start()

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    stop.set()

    ok = [r for r in results if r["error"] is None and r["ttft"] is not None]
    ttfts = sorted(r["ttft"] for r in ok)
    total_tokens = sum(r["tokens"] for r in ok)

    def pct(vals, p):
        if not vals:
            return 0.0
        return vals[min(len(vals) - 1, int(math.ceil(p / 100 * len(vals))) - 1)]

    if args.mock:
        # Simulated cache curve that saturates at the same knee as mock_request.
        peak["vllm:gpu_cache_usage_perc"] = min(1.0, concurrency / 7.0)
        peak["vllm:num_requests_waiting"] = max(0, concurrency - 6)
        peak["vllm:num_preemptions_total"] = max(0, (concurrency - 6) * 3)

    return {
        "concurrency": concurrency,
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "ttft_p50": round(statistics.median(ttfts), 4) if ttfts else 0.0,
        "ttft_p95": round(pct(ttfts, 95), 4),
        "wall_s": round(wall, 3),
        "output_tok_per_s": round(total_tokens / wall, 1) if wall > 0 else 0.0,
        "kv_cache_used": round(peak.get("vllm:gpu_cache_usage_perc", 0.0), 4),
        "queue_waiting": peak.get("vllm:num_requests_waiting", 0.0),
        "preemptions": peak.get("vllm:num_preemptions_total", 0.0),
    }


# ==========================================================================
# Output
# ==========================================================================


def find_knee(rows: list[dict]) -> int | None:
    """
    Locate the measured knee: the first concurrency level where p95 TTFT jumps
    by more than 50% over the previous level.

    A threshold rather than a derivative because the sweep has few points and
    real measurements are noisy; a second-difference method would fire on noise.
    """
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1]["ttft_p95"], rows[i]["ttft_p95"]
        if prev > 0 and cur > prev * 1.5:
            return rows[i]["concurrency"]
    return None


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_plot(rows: list[dict], path: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display in a container or over ssh
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    x = [r["concurrency"] for r in rows]
    knee = find_knee(rows)

    # WHY TWO STACKED PANELS RATHER THAN THREE LINES ON TWO AXES:
    # the first version put KV cache % (0-100) on the same right-hand axis as
    # throughput (which can be thousands of tok/s). The cache curve flattened
    # into an invisible line at the bottom of the chart. Percentages and
    # absolute rates do not share an axis legibly - so throughput gets its own
    # panel, and the two panels share the x-axis so the knee lines up visually
    # across both. Reading the alignment IS the point of the chart.
    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    # -- top panel: TTFT (left) and KV cache occupancy (right) -------------
    ax1.plot(x, [r["ttft_p95"] for r in rows], "o-", color="#d62728",
             linewidth=2.2, label="p95 TTFT (s)")
    ax1.plot(x, [r["ttft_p50"] for r in rows], "o:", color="#d62728",
             alpha=0.45, linewidth=1.4, label="p50 TTFT (s)")
    ax1.set_ylabel("time to first token (s)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, [r["kv_cache_used"] * 100 for r in rows], "s--", color="#1f77b4",
             linewidth=2.2, label="KV cache used (%)")
    ax2.plot(x, [r["queue_waiting"] for r in rows], "d-", color="#9467bd",
             alpha=0.8, linewidth=1.6, label="requests waiting")
    ax2.set_ylabel("KV cache used (%)  /  queue depth", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0, 105)
    ax2.axhline(100, color="#1f77b4", linestyle=":", alpha=0.4)

    # -- bottom panel: throughput -----------------------------------------
    ax3.plot(x, [r["output_tok_per_s"] for r in rows], "^-", color="#2ca02c",
             linewidth=2.2, label="output tokens/s")
    ax3.set_ylabel("throughput (tok/s)", color="#2ca02c")
    ax3.tick_params(axis="y", labelcolor="#2ca02c")
    ax3.set_xlabel("concurrent requests")
    ax3.grid(alpha=0.25)

    if knee:
        for ax in (ax1, ax3):
            ax.axvline(knee, color="black", linestyle=":", linewidth=2)
        ymax = max(r["ttft_p95"] for r in rows)
        ax1.annotate(
            f"knee at {knee} concurrent:\ncache is full, requests queue,\nTTFT breaks down",
            xy=(knee, ymax * 0.45),
            xytext=(knee + (max(x) - knee) * 0.12, ymax * 0.72),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9, fontweight="bold",
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)
    ax3.legend(loc="upper left", fontsize=8.5)
    ax1.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--metrics-url", default="http://localhost:8001/metrics")
    ap.add_argument("--levels", default="1,2,4,6,8,10,12,16")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--mock", action="store_true",
                    help="simulate; verifies this script with no GPU or server")
    ap.add_argument("--vram-gb", type=float, default=4.0)
    ap.add_argument("--geometry", default="qwen3-0.6b", choices=sorted(MODEL_GEOMETRY))
    ap.add_argument("--avg-seq-len", type=int, default=1024)
    ap.add_argument("--out-csv", default=str(ROOT / "data" / "kv_bench.csv"))
    ap.add_argument("--out-png", default=str(ROOT / "data" / "kv_bench.png"))
    args = ap.parse_args()

    # ---- STEP 1: predict, before measuring anything ---------------------
    geo = MODEL_GEOMETRY[args.geometry]
    pred = predict_knee(
        vram_gb=args.vram_gb, weight_gb=geo["weight_gb"], layers=geo["layers"],
        kv_heads=geo["kv_heads"], head_dim=geo["head_dim"],
        avg_seq_len=args.avg_seq_len,
    )
    print("=" * 66)
    print("PREDICTION (arithmetic, before any measurement)")
    print("=" * 66)
    if "error" in pred:
        print("  ", pred["error"])
    else:
        print(f"  model geometry      {args.geometry}: {geo['layers']} layers, "
              f"{geo['kv_heads']} kv-heads, head_dim {geo['head_dim']}")
        print(f"  KV bytes per token  {pred['bytes_per_token']} "
              f"({pred['kb_per_token']} KB)")
        print(f"  cache budget        {pred['cache_gb']} GB "
              f"({args.vram_gb} GB VRAM x 0.90 - {geo['weight_gb']} GB weights)")
        print(f"  max cached tokens   {pred['max_cached_tokens']:,}")
        print(f"  avg sequence        {pred['avg_seq_len']} tokens")
        print(f"  ==> PREDICTED KNEE  {pred['predicted_knee_concurrency']} "
              f"concurrent requests")
    print()

    if not args.mock:
        try:
            httpx.get(args.base_url.replace("/v1", "/health"), timeout=3.0)
        except Exception:
            print(f"WARNING: cannot reach {args.base_url}. Is vLLM running?")
            print("         Use --mock to verify this script without a server.\n")

    # ---- STEP 2: measure -------------------------------------------------
    levels = [int(x) for x in args.levels.split(",")]
    print("=" * 66)
    print("MEASUREMENT" + ("  (MOCK - simulated, not real hardware)" if args.mock else ""))
    print("=" * 66)
    print(f"{'conc':>5} {'ok':>4} {'fail':>5} {'p50 TTFT':>9} {'p95 TTFT':>9} "
          f"{'tok/s':>8} {'KV%':>6} {'wait':>5} {'preempt':>8}")
    print("-" * 66)

    rows = []
    for c in levels:
        r = sweep_level(args, c)
        rows.append(r)
        print(f"{r['concurrency']:>5} {r['ok']:>4} {r['failed']:>5} "
              f"{r['ttft_p50']:>9.3f} {r['ttft_p95']:>9.3f} "
              f"{r['output_tok_per_s']:>8.1f} {r['kv_cache_used']*100:>5.0f}% "
              f"{int(r['queue_waiting']):>5} {int(r['preemptions']):>8}")
        time.sleep(1.0)  # let the cache drain between levels

    # ---- STEP 3: compare prediction to measurement ----------------------
    measured = find_knee(rows)
    print()
    print("=" * 66)
    print("RESULT")
    print("=" * 66)
    print(f"  predicted knee  {pred.get('predicted_knee_concurrency', '?')} concurrent")
    print(f"  measured knee   {measured or 'not reached in this sweep'}")
    if measured and "predicted_knee_concurrency" in pred:
        p = pred["predicted_knee_concurrency"]
        err = abs(measured - p) / max(p, 1) * 100
        print(f"  error           {err:.0f}%")
        if err < 40:
            print("  -> prediction and measurement agree. The mental model is correct:")
            print("     TTFT degrades when the KV cache runs out of room, and where")
            print("     that happens is arithmetic, not magic.")
        else:
            print("  -> they disagree. Likely causes, in order: avg_seq_len is wrong")
            print("     (check actual prompt+output length), --max-model-len is capping")
            print("     the cache below your estimate, or the model geometry above is")
            print("     wrong (verify layers/kv_heads against the model's config.json).")

    write_csv(rows, Path(args.out_csv))
    print(f"\n  csv -> {args.out_csv}")
    ok = write_plot(rows, Path(args.out_png),
                    f"KV cache saturation - {args.model} on {args.vram_gb}GB"
                    + (" (MOCK)" if args.mock else ""))
    print(f"  png -> {args.out_png}" if ok else "  (matplotlib not installed; CSV only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
