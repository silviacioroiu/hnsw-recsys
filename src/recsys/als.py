"""Implicit-feedback matrix factorisation (ALS) with exact top-k search.

Tunes on the validation window, then refits on train + val and scores test.

Usage: python -m recsys.als [--split data/processed] [--out results]
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from recsys.evaluate import evaluate
from recsys.popularity import PopularityRecommender
from recsys.runinfo import collect_runinfo, set_seed


class ALSRecommender:
    """User and item embeddings; score(user, item) = dot product.

    Users with no liked item in the training data have no embedding and get
    `fallback`'s recommendations instead.
    """

    def __init__(self, factors: int = 64, regularization: float = 0.01,
                 alpha: float = 10.0, iterations: int = 15, seed: int = 0,
                 num_threads: int = 0, fallback=None):
        self.params = {"factors": factors, "regularization": regularization,
                       "alpha": alpha, "iterations": iterations}
        self.seed = seed
        self.num_threads = num_threads   # 0 = all cores
        self.fallback = fallback

    def fit(self, interactions: pd.DataFrame, min_rating: float = 4.0):
        liked = interactions[interactions["rating"] >= min_rating]
        # Map raw ids to matrix rows / columns 0..n-1.
        user_rows, self.user_ids = pd.factorize(liked["user_id"], sort=True)
        item_cols, self.item_ids = pd.factorize(liked["item_id"], sort=True)
        self.user_ids = np.asarray(self.user_ids)
        self.item_ids = np.asarray(self.item_ids)
        self.user_row = {int(user): row for row, user in enumerate(self.user_ids)}

        # Binary matrix: 1 where the user liked the item. `alpha` is the
        # confidence in a like, relative to 1 for "no interaction".
        user_items = csr_matrix(
            (np.ones(len(liked), dtype=np.float32), (user_rows, item_cols)),
            shape=(len(self.user_ids), len(self.item_ids)))
        # implicit parallelises over users itself; a multi-threaded BLAS inside
        # each of those threads oversubscribes the cores and slows training.
        with threadpool_limits(1, "blas"):
            model = AlternatingLeastSquares(
                **self.params, random_state=self.seed, num_threads=self.num_threads)
            model.fit(user_items, show_progress=False)
        self.user_factors = np.asarray(model.user_factors)
        self.item_factors = np.asarray(model.item_factors)
        return self

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        row = self.user_row.get(user_id)
        if row is None:
            return self.fallback.recommend(user_id, k, exclude) if self.fallback else []
        # Exact (brute-force) search: score every item. HNSW replaces this in M4.
        scores = self.item_factors @ self.user_factors[row]
        # Take k + |exclude| candidates so k remain after dropping excluded items;
        # argpartition finds them in O(n) instead of sorting all n scores.
        n = min(k + len(exclude), len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [item for item in self.item_ids[top].tolist() if item not in exclude][:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--artifacts", type=Path, default=Path("data/artifacts/als"))
    parser.add_argument("--factors", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--regularization", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--alpha", type=float, nargs="+", default=[10.0, 40.0])
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--fallback-window-days", type=int, default=30)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--threads", type=int, default=0, help="0 = all cores")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    train, val, test = (pd.read_parquet(args.split / f"{name}.parquet")
                        for name in ("train", "val", "test"))

    def fit_and_score(config: dict, history: pd.DataFrame, eval_df: pd.DataFrame):
        fallback = PopularityRecommender(args.fallback_window_days).fit(history, args.min_rating)
        model = ALSRecommender(**config, iterations=args.iterations, seed=args.seed,
                               num_threads=args.threads, fallback=fallback)
        start = time.perf_counter()
        model.fit(history, args.min_rating)
        fit_seconds = time.perf_counter() - start
        scores = evaluate(model, history, eval_df, min_rating=args.min_rating)
        return model, {**config, **scores, "fit_seconds": round(fit_seconds, 1)}

    # 1. Tune on validation only; the test window is never looked at here.
    grid = []
    for factors, regularization, alpha in itertools.product(
            args.factors, args.regularization, args.alpha):
        config = {"factors": factors, "regularization": regularization, "alpha": alpha}
        _, row = fit_and_score(config, train, val)
        grid.append(row)
        print(f"val  factors={factors:<4} reg={regularization:<5} alpha={alpha:<5} "
              f"recall@10={row['recall@10']:.4f}  ndcg@10={row['ndcg@10']:.4f}  "
              f"fit={row['fit_seconds']}s")

    # 2. Refit the best configuration on train + val and score test once.
    best = max(grid, key=lambda row: row["recall@10"])
    best_config = {name: best[name] for name in ("factors", "regularization", "alpha")}
    model, test_row = fit_and_score(best_config, pd.concat([train, val]), test)

    args.artifacts.mkdir(parents=True, exist_ok=True)
    for name in ("user_factors", "item_factors", "user_ids", "item_ids"):
        np.save(args.artifacts / f"{name}.npy", getattr(model, name))

    params = {"grid": {"factors": args.factors, "regularization": args.regularization,
                       "alpha": args.alpha},
              "iterations": args.iterations, "threads": args.threads,
              "fallback_window_days": args.fallback_window_days,
              "min_rating": args.min_rating, "selection_metric": "recall@10 on val",
              "split": json.loads((args.split / "split.json").read_text())}
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "m2_als.json"
    out_file.write_text(json.dumps({
        "runinfo": collect_runinfo(params, args.seed, packages=("scipy", "implicit")),
        "validation_grid": grid,
        "test": {"model": "als", "eval_window": "test", **test_row},
    }, indent=2))

    print(f"\nBest on val: {best_config}")
    print("test " + "  ".join(f"{m}={test_row[m]:.4f}"
                              for m in ("recall@10", "ndcg@10", "recall@50", "ndcg@50"))
          + f"  users={test_row['n_users_evaluated']}")
    print(f"Wrote {out_file} and embeddings in {args.artifacts}")


if __name__ == "__main__":
    main()
