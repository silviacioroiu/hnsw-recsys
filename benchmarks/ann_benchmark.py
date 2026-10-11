"""Recall@10 versus latency: this HNSW index against exact search and FAISS.

Items are the ALS film embeddings; queries are user embeddings; the
similarity is the inner product. "Recall" here means overlap with the exact
top 10, not recommendation quality.

Usage: python benchmarks/ann_benchmark.py [--artifacts data/artifacts/als]
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

from recsys._hnsw import Index
from recsys.runinfo import collect_runinfo, set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

K = 10
WARMUP = 50


def measure(search, queries: np.ndarray, truth: np.ndarray) -> dict:
    """Recall against `truth` and per-query latency of `search(query) -> ids`."""
    for query in queries[:WARMUP]:
        search(query)
    latencies_ms, hits = [], 0
    for query, exact in zip(queries, truth):
        start = time.perf_counter_ns()
        ids = search(query)
        latencies_ms.append((time.perf_counter_ns() - start) / 1e6)
        hits += len(set(np.asarray(ids).tolist()) & set(exact.tolist()))
    return {
        "recall@10": hits / (len(queries) * K),
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
    }


def plot(rows: list[dict], Ms: list[int], out_file: Path) -> None:
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
    series = {"hnsw (this repo)": "#2a78d6", "faiss hnsw": "#eb6834"}
    exact = next(r for r in rows if r["method"] == "faiss flat (exact)")

    fig, axes = plt.subplots(1, len(Ms), figsize=(4.2 * len(Ms), 3.9), sharey=True,
                             facecolor=surface)
    for ax, M in zip(axes, Ms):
        ax.set_facecolor(surface)
        for method, color in series.items():
            points = sorted((r for r in rows if r["method"] == method and r["M"] == M),
                            key=lambda r: r["ef"])
            ax.plot([p["latency_ms_p50"] for p in points], [p["recall@10"] for p in points],
                    color=color, linewidth=2, marker="o", markersize=5, label=method)
        # Exact search is the reference: recall 1.0 at this latency.
        ax.axvline(exact["latency_ms_p50"], color=muted, linewidth=1, linestyle="--")
        ax.text(exact["latency_ms_p50"], 0.02, " exact search", color=muted, fontsize=8,
                rotation=90, va="bottom", ha="right", transform=ax.get_xaxis_transform())
        ax.set_xscale("log")
        ax.set_title(f"M = {M}", color=ink, fontsize=11, loc="left")
        ax.set_xlabel("median latency per query (ms, log scale)", color=ink, fontsize=9)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.tick_params(colors=muted, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("recall@10 against exact search", color=ink, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=ink, loc="lower right")
    fig.suptitle("Recall versus latency; each point is one ef, increasing left to right",
                 color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_file, dpi=160, facecolor=surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("data/artifacts/als"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--M", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--ef", type=int, nargs="+", default=[10, 20, 50, 100, 200, 500])
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    items = np.ascontiguousarray(np.load(args.artifacts / "item_factors.npy"), dtype=np.float32)
    users = np.load(args.artifacts / "user_factors.npy").astype(np.float32)
    rng = np.random.default_rng(args.seed)
    queries = users[rng.choice(len(users), size=args.queries, replace=False)]
    n_items, dim = items.shape

    params = {"k": K, "n_items": n_items, "dim": dim, "n_queries": args.queries,
              "metric": "inner product", "M": args.M, "ef": args.ef,
              "ef_construction": args.ef_construction, "threads": 1, "warmup_queries": WARMUP,
              "timing": "one query per call from Python, perf_counter_ns"}
    runinfo = collect_runinfo(params, args.seed, packages=("faiss-cpu", "pybind11"))

    # Everything runs on one thread so the methods are compared like for like.
    faiss.omp_set_num_threads(1)
    with threadpool_limits(1):
        truth = np.argsort(-(queries @ items.T), axis=1)[:, :K]
        rows = []

        def record(method: str, search, **extra) -> None:
            row = {"method": method, **extra, **measure(search, queries, truth)}
            rows.append(row)
            print(f"{method:<20} M={str(extra.get('M', '-')):<3} ef={str(extra.get('ef', '-')):<4}"
                  f" recall@10={row['recall@10']:.4f}  p50={row['latency_ms_p50']:.4f}ms"
                  f"  p95={row['latency_ms_p95']:.4f}ms")

        def numpy_exact(query):
            scores = items @ query
            top = np.argpartition(-scores, K - 1)[:K]
            return top[np.argsort(-scores[top])]

        record("numpy (exact)", numpy_exact)

        flat = faiss.IndexFlatIP(dim)
        flat.add(items)
        record("faiss flat (exact)", lambda q: flat.search(q[None, :], K)[1][0])

        for M in args.M:
            start = time.perf_counter()
            ours = Index(dim=dim, metric="ip", M=M, ef_construction=args.ef_construction,
                         seed=args.seed)
            ours.add(items)
            ours_build = round(time.perf_counter() - start, 2)

            start = time.perf_counter()
            theirs = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            theirs.hnsw.efConstruction = args.ef_construction
            theirs.add(items)
            theirs_build = round(time.perf_counter() - start, 2)

            for ef in args.ef:
                record("hnsw (this repo)", lambda q: ours.search(q, K, ef)[0],
                       M=M, ef=ef, build_seconds=ours_build)
                theirs.hnsw.efSearch = ef
                record("faiss hnsw", lambda q: theirs.search(q[None, :], K)[1][0],
                       M=M, ef=ef, build_seconds=theirs_build)

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "m4_ann_benchmark.json"
    out_file.write_text(json.dumps({"runinfo": runinfo, "results": rows}, indent=2))
    plot_file = args.out / "m4_recall_vs_latency.png"
    plot(rows, args.M, plot_file)
    print(f"Wrote {out_file} and {plot_file}")


if __name__ == "__main__":
    main()
