import numpy as np
import pandas as pd
import pytest

from recsys.als import ALSRecommender
from recsys.popularity import PopularityRecommender

GROUP_A_ITEMS = set(range(0, 20))
GROUP_B_ITEMS = set(range(100, 120))


@pytest.fixture
def two_taste_groups() -> pd.DataFrame:
    """Users 0-19 like only items 0-19; users 20-39 like only items 100-119."""
    rng = np.random.default_rng(0)
    rows = []
    for user in range(40):
        items = sorted(GROUP_A_ITEMS if user < 20 else GROUP_B_ITEMS)
        for item in rng.choice(items, size=12, replace=False):
            rows.append((user, int(item), 5.0, len(rows)))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def _fit(data, **kwargs):
    # Two factors are enough for two groups; more would fit noise on 480 rows.
    return ALSRecommender(factors=2, iterations=15, seed=0, num_threads=1, **kwargs).fit(data)


def test_recommends_within_the_users_taste_group(two_taste_groups):
    model = _fit(two_taste_groups)
    seen = two_taste_groups.groupby("user_id")["item_id"].agg(set)
    for user in range(40):
        group = GROUP_A_ITEMS if user < 20 else GROUP_B_ITEMS
        recommended = model.recommend(user, 5, seen[user])
        assert len(recommended) == 5
        assert set(recommended) <= group - seen[user]


def test_results_are_ordered_by_score(two_taste_groups):
    model = _fit(two_taste_groups)
    recommended = model.recommend(0, 10, set())
    position = {int(item): i for i, item in enumerate(model.item_ids)}
    scores = [model.item_factors[position[item]] @ model.user_factors[model.user_row[0]]
              for item in recommended]
    assert scores == sorted(scores, reverse=True)


def test_same_seed_gives_same_embeddings(two_taste_groups):
    first, second = _fit(two_taste_groups), _fit(two_taste_groups)
    np.testing.assert_allclose(first.user_factors, second.user_factors)
    np.testing.assert_allclose(first.item_factors, second.item_factors)


def test_user_without_likes_gets_the_fallback(two_taste_groups):
    # User 99 rated one film badly: history, but nothing to learn a taste from.
    data = pd.concat([two_taste_groups, pd.DataFrame(
        [(99, 0, 1.0, 10_000)], columns=two_taste_groups.columns)])
    fallback = PopularityRecommender().fit(data)
    model = _fit(data, fallback=fallback)
    assert 99 not in model.user_row
    assert model.recommend(99, 5, {0}) == fallback.recommend(99, 5, {0})
    assert 0 not in model.recommend(99, 5, {0})
