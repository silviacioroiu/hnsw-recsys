"""The full pipeline as the service runs it: HNSW retrieval, then re-ranking.

Exports compact serving files so the service never loads the ratings, then
reloads them and scores the served pipeline on the test window, to confirm
it gives the quality measured in M5.

Usage: python -m recsys.serving [--split data/processed] [--out results]
"""

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from recsys._hnsw import Index
from recsys.als import ALSRecommender
from recsys.data import load_movies
from recsys.evaluate import evaluate
from recsys.features import FeatureBuilder
from recsys.runinfo import collect_runinfo, set_seed

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200


class ServingPipeline:
    """user id -> top-k item ids, from files under one artifacts directory:

    als/      user and item embeddings            (written by recsys.als)
    ranker/   LightGBM model                      (written by recsys.ranking)
    serving/  features, seen items, titles        (written by `export`)
    """

    def __init__(self, builder: FeatureBuilder, model: lgb.Booster, seen: csr_matrix,
                 seen_users: np.ndarray, titles: np.ndarray, seed: int = 0):
        self.builder = builder
        self.model = model
        self.seen = seen                 # user row x item row, 1 = already rated
        self.seen_users = seen_users
        self.seen_row = {int(user): row for row, user in enumerate(seen_users)}
        self.titles = titles             # aligned with builder.item_ids
        # Built at start-up (a few seconds for ~30k items) rather than saved:
        # the index has no serialisation yet.
        index = Index(dim=builder.als.item_factors.shape[1], metric="ip", M=HNSW_M,
                      ef_construction=HNSW_EF_CONSTRUCTION, seed=seed)
        index.add(builder.als.item_factors)
        builder.use_index(index)

    @staticmethod
    def export(directory: Path, history: pd.DataFrame, movies: pd.DataFrame,
               builder: FeatureBuilder) -> None:
        """Write serving/ from the history the embeddings were fitted on."""
        serving = directory / "serving"
        builder.save(serving)
        # Which items each user has rated, as rows of the item embedding
        # matrix. Items without an embedding can never be recommended, so they
        # do not need to be remembered.
        items = pd.Index(builder.item_ids)
        user_codes, users = pd.factorize(history["user_id"])
        item_rows = items.get_indexer(history["item_id"])
        known = item_rows >= 0
        seen = csr_matrix((np.ones(known.sum(), dtype=np.bool_),
                           (user_codes[known], item_rows[known])),
                          shape=(len(users), len(items)))
        save_npz(serving / "seen.npz", seen)
        np.save(serving / "seen_users.npy", np.asarray(users))
        titles = movies.set_index("item_id")["title"].reindex(items).fillna("")
        np.save(serving / "titles.npy", titles.to_numpy(dtype=str))

    @classmethod
    def load(cls, directory: Path, seed: int = 0) -> "ServingPipeline":
        serving = directory / "serving"
        builder = FeatureBuilder.load(serving, ALSRecommender.load(directory / "als"))
        model = lgb.Booster(model_file=str(directory / "ranker" / "model.txt"))
        return cls(builder, model, load_npz(serving / "seen.npz").tocsr(),
                   np.load(serving / "seen_users.npy"), np.load(serving / "titles.npy"), seed)

    def knows(self, user_id: int) -> bool:
        return user_id in self.seen_row

    def recommend_rows(self, user_id: int, k: int) -> np.ndarray:
        """Row numbers (into item_ids / titles) of the top k, best first."""
        row = self.seen_row.get(user_id)
        if row is None:
            seen_rows = np.empty(0, dtype=np.int64)
        else:
            seen_rows = self.seen.indices[self.seen.indptr[row]:self.seen.indptr[row + 1]]
        items, features = self.builder.candidates_from_rows(user_id, seen_rows)
        if len(items) == 0:
            return np.empty(0, dtype=np.int64)
        # One thread per request: the server already handles requests in parallel.
        scores = self.model.predict(features, num_threads=1)
        top_items = items[np.argsort(-scores, kind="stable")][:k]
        return np.array([self.builder.item_row[int(item)] for item in top_items], dtype=np.int64)

    def recommend(self, user_id: int, k: int, exclude: set[int] | None = None) -> list[int]:
        """Matches the Recommender protocol; `exclude` is ignored because the
        pipeline keeps its own record of what each user has seen."""
        return self.builder.item_ids[self.recommend_rows(user_id, k)].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/ml-25m"))
    parser.add_argument("--split", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts", type=Path, default=Path("data/artifacts"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    m5 = json.loads((args.out / "m5_ranking.json").read_text())
    candidates = {name: m5["runinfo"]["params"][name]
                  for name in ("n_als", "n_popular", "window_days")}
    params = {**candidates, "min_rating": args.min_rating, "hnsw_M": HNSW_M,
              "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
              "reference": m5["test"][1]}
    runinfo = collect_runinfo(params, args.seed, packages=("lightgbm", "pybind11", "scipy"))

    # The service answers "as of the test cutoff": same history as the M5 score.
    history = pd.concat(pd.read_parquet(args.split / f"{name}.parquet")
                        for name in ("train", "val"))
    movies = load_movies(args.raw / "movies.csv")
    builder = FeatureBuilder(history, movies, ALSRecommender.load(args.artifacts / "als"),
                             min_rating=args.min_rating, **candidates)
    ServingPipeline.export(args.artifacts, history, movies, builder)

    # Score what was written to disk, not the objects still in memory.
    start = time.perf_counter()
    pipeline = ServingPipeline.load(args.artifacts, args.seed)
    load_seconds = round(time.perf_counter() - start, 1)
    test = pd.read_parquet(args.split / "test.parquet")
    start = time.perf_counter()
    scores = evaluate(pipeline, history, test, min_rating=args.min_rating)
    ms_per_user = 1000 * (time.perf_counter() - start) / scores["n_users_evaluated"]

    out_file = args.out / "m6_serving_quality.json"
    out_file.write_text(json.dumps({
        "runinfo": runinfo, "load_seconds": load_seconds,
        "test": {"stage": "served pipeline (HNSW + LightGBM)", **scores,
                 "eval_ms_per_user": round(ms_per_user, 3)},
    }, indent=2))

    metrics = ("recall@10", "ndcg@10", "recall@50", "ndcg@50")
    print(f"Exported to {args.artifacts / 'serving'}; pipeline loads in {load_seconds}s")
    print("served (HNSW)  " + "  ".join(f"{m}={scores[m]:.4f}" for m in metrics)
          + f"  {ms_per_user:.2f} ms/user")
    print("M5 (exact)     " + "  ".join(f"{m}={m5['test'][1][m]:.4f}" for m in metrics))
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
