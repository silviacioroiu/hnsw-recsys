import math

import pandas as pd
import pytest

from recsys.evaluate import evaluate
from recsys.popularity import PopularityRecommender
from recsys.split import temporal_split


class FixedRecommender:
    """Returns the same list for everyone and records what it was asked."""

    def __init__(self, items):
        self.items = items
        self.calls = {}

    def recommend(self, user_id, k, exclude):
        self.calls[user_id] = (k, exclude)
        return [item for item in self.items if item not in exclude][:k]


def _interactions(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_scores_only_users_with_history_and_positives():
    history = _interactions([(1, 10, 5.0, 1), (1, 11, 2.0, 2), (2, 10, 4.0, 3)])
    eval_df = _interactions([
        (1, 20, 5.0, 10),   # user 1: one positive, recommended at rank 1
        (2, 21, 2.0, 11),   # user 2: history but no positive -> not scored
        (3, 20, 5.0, 12),   # user 3: positive but no history -> not scored
    ])
    model = FixedRecommender([20, 21, 22])
    result = evaluate(model, history, eval_df, ks=(1, 2))

    assert result["n_users_evaluated"] == 1
    assert result["n_users_without_history"] == 1
    assert result["recall@1"] == 1.0
    assert result["ndcg@2"] == 1.0
    # Asked once, for the largest k, excluding everything user 1 rated.
    assert model.calls == {1: (2, {10, 11})}


def test_metrics_are_averaged_over_users():
    history = _interactions([(1, 10, 5.0, 1), (2, 10, 5.0, 2)])
    eval_df = _interactions([(1, 20, 5.0, 10), (2, 30, 5.0, 11)])
    result = evaluate(FixedRecommender([20]), history, eval_df, ks=(1,))
    assert result["recall@1"] == 0.5   # user 1 hit, user 2 miss


def test_positives_on_unknown_items_are_reported():
    history = _interactions([(1, 10, 5.0, 1), (2, 20, 5.0, 2)])
    # Item 20 exists in history (rated by user 2); item 99 does not.
    eval_df = _interactions([(1, 20, 5.0, 10), (1, 99, 5.0, 11)])
    result = evaluate(FixedRecommender([20]), history, eval_df, ks=(1,))
    assert result["frac_positives_on_unknown_items"] == 0.5
    assert result["recall@1"] == 0.5


def test_no_evaluable_users_is_an_error():
    history = _interactions([(1, 10, 5.0, 1)])
    eval_df = _interactions([(2, 20, 5.0, 10)])
    with pytest.raises(ValueError):
        evaluate(FixedRecommender([20]), history, eval_df)


def test_baseline_never_recommends_seen_items_and_is_deterministic(ratings):
    train, val, _, _ = temporal_split(ratings)
    model = PopularityRecommender().fit(train)

    seen = train.groupby("user_id")["item_id"].agg(set)
    for user, items in seen.items():
        assert not items & set(model.recommend(user, 50, items))

    first = evaluate(model, train, val)
    second = evaluate(PopularityRecommender().fit(train), train, val)
    assert first == second
    assert 0.0 <= first["recall@10"] <= first["recall@50"] <= 1.0
    assert first["n_users_evaluated"] > 0
    assert not math.isnan(first["ndcg@10"])
