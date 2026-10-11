"""ALS recommendations retrieved with the HNSW index instead of exact search.

Scores the same embeddings both ways on the test window, so the only
difference between the rows is the search.

Usage: python -m recsys.ann [--split data/processed] [--out results]
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from recsys._hnsw import Index
from recsys.als import ALSRecommender
from recsys.evaluate import evaluate
from recsys.popularity import PopularityRecommender
from recsys.runinfo import collect_runinfo, set_seed


class HNSWRecommender:
    """Wraps a fitted ALSRecommender; only the item search changes."""

    def __init__(self, als: ALSRecommender, M: int = 16, ef_construction: int = 200,
                 ef: int = 100, seed: int = 0):
        self.als = als
        self.ef = ef
        self.index = Index(dim=als.item_factors.shape[1], metric="ip", M=M,
                           ef_construction=ef_construction, seed=seed)
        self.index.add(als.item_factors)

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        row = self.als.user_row.get(user_id)
        if row is None:
            return self.als.recommend(user_id, k, exclude)   # popularity fallback
        # The index cannot skip items during the search, so ask for k + |exclude|
        # and filter afterwards. For a user who has rated thousands of films
        # this makes the search much wider than k; see the README trade-offs.
        ids, _ = self.index.search(self.als.user_factors[row], k + len(exclude), self.ef)
        return [item for item in self.als.item_ids[ids].tolist() if item not in exclude][:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts", type=Path, default=Path("data/artifacts/als"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef", type=int, nargs="+", default=[10, 50, 100, 200])
    parser.add_argument("--fallback-window-days", type=int, default=30)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    params = {"M": args.M, "ef_construction": args.ef_construction, "ef": args.ef,
              "fallback_window_days": args.fallback_window_days,
              "min_rating": args.min_rating,
              "embeddings": json.loads(Path(args.out / "m2_als.json").read_text())["test"]}
    runinfo = collect_runinfo(params, args.seed, packages=("pybind11",))

    # The saved embeddings were fitted on train + val, so that is the history.
    history = pd.concat(pd.read_parquet(args.split / f"{name}.parquet")
                        for name in ("train", "val"))
    test = pd.read_parquet(args.split / "test.parquet")
    fallback = PopularityRecommender(args.fallback_window_days).fit(history, args.min_rating)
    als = ALSRecommender.load(args.artifacts, fallback=fallback)

    def score(name: str, recommender, **extra) -> dict:
        start = time.perf_counter()
        scores = evaluate(recommender, history, test, min_rating=args.min_rating)
        seconds = time.perf_counter() - start
        # Includes the per-user filtering in Python, not only the search.
        ms_per_user = 1000 * seconds / scores["n_users_evaluated"]
        return {"search": name, **extra, **scores, "eval_ms_per_user": round(ms_per_user, 3)}

    rows = [score("exact", als)]
    start = time.perf_counter()
    hnsw = HNSWRecommender(als, args.M, args.ef_construction, seed=args.seed)
    build_seconds = round(time.perf_counter() - start, 1)
    for ef in args.ef:
        hnsw.ef = ef
        rows.append(score("hnsw", hnsw, ef=ef))

    out_file = args.out / "m4_retrieval.json"
    out_file.write_text(json.dumps(
        {"runinfo": runinfo, "n_items": len(als.item_ids),
         "hnsw_build_seconds": build_seconds, "results": rows}, indent=2))

    metrics = ["recall@10", "ndcg@10", "recall@50", "ndcg@50"]
    print(f"{len(als.item_ids)} items, HNSW build {build_seconds}s")
    print(f"{'search':<8}{'ef':>5}" + "".join(f"{m:>11}" for m in metrics) + f"{'ms/user':>10}")
    for row in rows:
        print(f"{row['search']:<8}{row.get('ef', '-'):>5}"
              + "".join(f"{row[m]:>11.4f}" for m in metrics)
              + f"{row['eval_ms_per_user']:>10.3f}")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
