"""One evaluation harness shared by every stage, so results are comparable."""

from typing import Protocol

import pandas as pd

from recsys.metrics import ndcg_at_k, recall_at_k


class Recommender(Protocol):
    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        """Top-k item ids for the user, best first, none of them in `exclude`."""
        ...


def evaluate(
    recommender: Recommender,
    history: pd.DataFrame,
    eval_df: pd.DataFrame,
    ks: tuple[int, ...] = (10, 50),
    min_rating: float = 4.0,
) -> dict:
    """Mean Recall@K and NDCG@K over the evaluable users.

    history: every interaction before the evaluation window.
    eval_df: the interactions inside the evaluation window.

    A user is evaluable if they have history and at least one positive
    (rating >= min_rating) in the window. Users without history are counted
    and reported, not scored: none of our models can personalise for them.
    """
    positives = (eval_df[eval_df["rating"] >= min_rating]
                 .groupby("user_id")["item_id"].agg(set))
    # Exclude everything the user rated, not only what they liked: a film
    # they gave 2 stars is still a film they have already seen.
    seen = (history[history["user_id"].isin(positives.index)]
            .groupby("user_id")["item_id"].agg(set))
    users = positives.index.intersection(seen.index)
    if len(users) == 0:
        raise ValueError("No evaluable users: nobody has both history and a positive")
    known_items = set(history["item_id"].unique())

    max_k = max(ks)
    totals = {f"{name}@{k}": 0.0 for k in ks for name in ("recall", "ndcg")}
    n_positives = n_unknown_item = 0
    for user in users:
        relevant = positives[user]
        recommended = recommender.recommend(int(user), max_k, seen[user])
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(recommended, relevant, k)
            totals[f"ndcg@{k}"] += ndcg_at_k(recommended, relevant, k)
        n_positives += len(relevant)
        n_unknown_item += len(relevant - known_items)

    n_users = len(users)
    return {
        **{name: total / n_users for name, total in totals.items()},
        "n_users_evaluated": n_users,
        "n_users_without_history": len(positives) - n_users,
        # Positives on items released after the cutoff. They stay in the
        # ground truth, so this fraction is recall no model here can reach.
        "frac_positives_on_unknown_items": n_unknown_item / n_positives,
    }
