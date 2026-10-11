"""Most-popular baselines: all-time and recent-window.

Usage: python -m recsys.popularity [--split data/processed] [--out results]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.evaluate import evaluate
from recsys.runinfo import collect_runinfo, set_seed

SECONDS_PER_DAY = 86_400


class PopularityRecommender:
    """Recommends the items with the most positive ratings, same for everyone.

    window_days=None counts all of the history; otherwise only the last
    window_days before the end of the history.
    """

    def __init__(self, window_days: int | None = None):
        self.window_days = window_days
        self.ranked_items: list[int] = []

    def fit(self, interactions: pd.DataFrame, min_rating: float = 4.0):
        liked = interactions[interactions["rating"] >= min_rating]
        if self.window_days is not None:
            start = interactions["timestamp"].max() - self.window_days * SECONDS_PER_DAY
            liked = liked[liked["timestamp"] >= start]
        counts = liked.groupby("item_id").size()
        # Most liked first; ties broken by item id so the ranking is deterministic.
        order = np.lexsort((counts.index.to_numpy(), -counts.to_numpy()))
        self.ranked_items = counts.index.to_numpy()[order].tolist()
        return self

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        recommended = []
        for item in self.ranked_items:
            if item not in exclude:
                recommended.append(item)
                if len(recommended) == k:
                    break
        return recommended


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    train, val, test = (pd.read_parquet(args.split / f"{name}.parquet")
                        for name in ("train", "val", "test"))
    # For the test window the model is refit on train + val, as it would be
    # in production: it may use everything that happened before the cutoff.
    windows = {"val": (train, val), "test": (pd.concat([train, val]), test)}
    models = {"popular_all_time": None, f"popular_last_{args.window_days}d": args.window_days}

    rows = []
    for window, (history, eval_df) in windows.items():
        for model, window_days in models.items():
            recommender = PopularityRecommender(window_days).fit(history, args.min_rating)
            scores = evaluate(recommender, history, eval_df, min_rating=args.min_rating)
            rows.append({"model": model, "eval_window": window, **scores})

    params = {"window_days": args.window_days, "min_rating": args.min_rating,
              "split": json.loads((args.split / "split.json").read_text())}
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "m1_popularity.json"
    out_file.write_text(json.dumps(
        {"runinfo": collect_runinfo(params, args.seed), "results": rows}, indent=2))

    metrics = ["recall@10", "ndcg@10", "recall@50", "ndcg@50"]
    print(f"{'model':<20}{'window':<8}" + "".join(f"{m:>11}" for m in metrics) + f"{'users':>9}")
    for row in rows:
        print(f"{row['model']:<20}{row['eval_window']:<8}"
              + "".join(f"{row[m]:>11.4f}" for m in metrics)
              + f"{row['n_users_evaluated']:>9}")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
