"""LightGBM re-ranking of the ALS + popularity candidates.

The ranker learns on the validation window from features built on the train
window only, then is scored once on the test window with features rebuilt
from train + val.

Usage: python -m recsys.ranking [--split data/processed] [--out results]
"""

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from recsys.als import ALSRecommender
from recsys.data import load_movies
from recsys.evaluate import evaluate
from recsys.features import FEATURES, FeatureBuilder
from recsys.runinfo import collect_runinfo, set_seed


class CandidateRecommender:
    """The candidates in their original order; used to measure their recall."""

    def __init__(self, builder: FeatureBuilder):
        self.builder = builder

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        return self.builder.candidates(user_id, exclude)[0][:k].tolist()


class RankedRecommender:
    """Candidates re-ordered by the ranker's score."""

    def __init__(self, builder: FeatureBuilder, model: lgb.Booster):
        self.builder = builder
        self.model = model

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        items, features = self.builder.candidates(user_id, exclude)
        if len(items) == 0:
            return []
        scores = self.model.predict(features)
        return items[np.argsort(-scores, kind="stable")][:k].tolist()


def build_training_set(builder: FeatureBuilder, history: pd.DataFrame,
                       labels_df: pd.DataFrame, min_rating: float = 4.0):
    """One group of candidate rows per user; label 1 if liked in `labels_df`.

    Returns a list of (features, labels) per user. Users whose candidates
    contain none of their liked items are skipped: with nothing to rank
    above anything else, they carry no signal for a ranking loss.
    """
    positives = (labels_df[labels_df["rating"] >= min_rating]
                 .groupby("user_id")["item_id"].agg(set))
    seen = (history[history["user_id"].isin(positives.index)]
            .groupby("user_id")["item_id"].agg(set))
    groups = []
    for user in positives.index.intersection(seen.index):
        items, features = builder.candidates(int(user), seen[user])
        labels = np.isin(items, list(positives[user])).astype(np.int8)
        if labels.any():
            groups.append((features, labels))
    return groups


def train_ranker(groups: list, seed: int = 0, holdout: float = 0.2,
                 threads: int = 0, **params) -> lgb.Booster:
    """LambdaRank on the user groups; a share of users decides when to stop."""
    order = np.random.default_rng(seed).permutation(len(groups))
    n_holdout = max(1, int(len(groups) * holdout))

    def dataset(indices, reference=None) -> lgb.Dataset:
        return lgb.Dataset(
            np.vstack([groups[i][0] for i in indices]),
            label=np.concatenate([groups[i][1] for i in indices]),
            group=[len(groups[i][1]) for i in indices],
            feature_name=FEATURES, reference=reference)

    train_set = dataset(order[n_holdout:])
    holdout_set = dataset(order[:n_holdout], reference=train_set)
    settings = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
                "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
                "seed": seed, "deterministic": True, "force_col_wise": True,
                "num_threads": threads, "verbose": -1, **params}
    return lgb.train(settings, train_set, num_boost_round=1000, valid_sets=[holdout_set],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/ml-25m"))
    parser.add_argument("--split", type=Path, default=Path("data/processed"))
    parser.add_argument("--als-artifacts", type=Path, default=Path("data/artifacts/als"))
    parser.add_argument("--artifacts", type=Path, default=Path("data/artifacts/ranker"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--n-als", type=int, default=500)
    parser.add_argument("--n-popular", type=int, default=100)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--threads", type=int, default=0, help="0 = all cores")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    m2 = json.loads((args.out / "m2_als.json").read_text())
    als_config = {name: m2["test"][name] for name in ("factors", "regularization", "alpha")}
    als_config["iterations"] = m2["runinfo"]["params"]["iterations"]
    params = {"n_als": args.n_als, "n_popular": args.n_popular,
              "window_days": args.window_days, "min_rating": args.min_rating,
              "threads": args.threads, "als": als_config, "features": FEATURES}
    runinfo = collect_runinfo(params, args.seed, packages=("lightgbm", "implicit", "scipy"))

    train, val, test = (pd.read_parquet(args.split / f"{name}.parquet")
                        for name in ("train", "val", "test"))
    movies = load_movies(args.raw / "movies.csv")
    builder_args = {"min_rating": args.min_rating, "n_als": args.n_als,
                    "n_popular": args.n_popular, "window_days": args.window_days}

    # 1. Learn on validation. The embeddings saved by M2 were fitted on
    #    train + val, so they have seen these labels; refit on train alone.
    als_train = ALSRecommender(**als_config, seed=args.seed, num_threads=args.threads).fit(
        train, args.min_rating)
    groups = build_training_set(
        FeatureBuilder(train, movies, als_train, **builder_args), train, val, args.min_rating)
    start = time.perf_counter()
    model = train_ranker(groups, seed=args.seed, threads=args.threads)
    train_seconds = round(time.perf_counter() - start, 1)
    n_rows = sum(len(labels) for _, labels in groups)
    n_positive = int(sum(labels.sum() for _, labels in groups))
    print(f"Trained on {len(groups)} users, {n_rows} rows, {n_positive} positives; "
          f"{model.best_iteration} trees in {train_seconds}s")

    # 2. Score test once, with everything rebuilt from train + val.
    history = pd.concat([train, val])
    builder = FeatureBuilder(history, movies, ALSRecommender.load(args.als_artifacts),
                             **builder_args)
    n_candidates = args.n_als + args.n_popular

    def score(stage: str, recommender, ks) -> dict:
        start = time.perf_counter()
        scores = evaluate(recommender, history, test, ks=ks, min_rating=args.min_rating)
        ms_per_user = 1000 * (time.perf_counter() - start) / scores["n_users_evaluated"]
        return {"stage": stage, **scores, "eval_ms_per_user": round(ms_per_user, 3)}

    # Recall at the full candidate count is the ceiling the ranker works under.
    candidates = score("candidates (ALS order, then popular)", CandidateRecommender(builder),
                       (10, 50, n_candidates))
    reranked = score("lightgbm re-ranked", RankedRecommender(builder, model), (10, 50))

    args.artifacts.mkdir(parents=True, exist_ok=True)
    model.save_model(args.artifacts / "model.txt")
    gain = model.feature_importance(importance_type="gain")
    importance = {name: round(float(share), 4) for name, share in
                  sorted(zip(FEATURES, gain / gain.sum()), key=lambda pair: -pair[1])}
    out_file = args.out / "m5_ranking.json"
    out_file.write_text(json.dumps({
        "runinfo": runinfo,
        "training": {"users": len(groups), "rows": n_rows, "positives": n_positive,
                     "trees": model.best_iteration, "seconds": train_seconds,
                     "holdout_ndcg@10": model.best_score["valid_0"]["ndcg@10"]},
        "feature_importance_gain": importance,
        "test": [candidates, reranked],
    }, indent=2))

    print(f"Candidate recall@{n_candidates}: {candidates[f'recall@{n_candidates}']:.4f}")
    for row in (candidates, reranked):
        print(f"{row['stage']:<38}" + "  ".join(
            f"{m}={row[m]:.4f}" for m in ("recall@10", "ndcg@10", "recall@50", "ndcg@50")))
    print("Top features by gain: " + ", ".join(
        f"{name} {share:.0%}" for name, share in list(importance.items())[:5]))
    print(f"Wrote {out_file} and {args.artifacts / 'model.txt'}")


if __name__ == "__main__":
    main()
