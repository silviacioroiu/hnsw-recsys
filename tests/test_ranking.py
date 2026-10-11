import numpy as np
import pandas as pd
import pytest

from recsys.als import ALSRecommender
from recsys.evaluate import evaluate
from recsys.features import FEATURES, FeatureBuilder
from recsys.ranking import (CandidateRecommender, RankedRecommender, build_training_set,
                            train_ranker)
from recsys.split import temporal_split


@pytest.fixture
def movies() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    genres = ["Action", "Comedy", "Drama", "Horror"]
    return pd.DataFrame({
        "item_id": np.arange(1, 300),
        "title": [f"Film {i} ({1990 + i % 30})" for i in range(1, 300)],
        "genres": ["|".join(rng.choice(genres, size=rng.integers(1, 3), replace=False))
                   for _ in range(1, 300)],
        "year": (1990 + np.arange(1, 300) % 30).astype("float32"),
    })


@pytest.fixture
def setup(ratings, movies):
    train, val, _, _ = temporal_split(ratings)
    als = ALSRecommender(factors=8, iterations=5, seed=0, num_threads=1).fit(train)
    builder = FeatureBuilder(train, movies, als, n_als=50, n_popular=20)
    return train, val, als, builder


def test_candidates_are_unique_unseen_and_have_one_row_of_features_each(setup):
    train, _, als, builder = setup
    seen = train.groupby("user_id")["item_id"].agg(set)
    for user in list(als.user_row)[:20]:
        items, features = builder.candidates(user, seen[user])
        assert features.shape == (len(items), len(FEATURES))
        assert features.dtype == np.float32
        assert len(set(items.tolist())) == len(items)
        assert not seen[user] & set(items.tolist())
        assert 50 <= len(items) <= 70


def test_als_candidates_come_first_in_score_order(setup):
    _, _, als, builder = setup
    user = next(iter(als.user_row))
    items, features = builder.candidates(user, set())
    scores, ranks = features[:, FEATURES.index("als_score")], features[:, FEATURES.index("als_rank")]
    assert items[:50].tolist() == als.recommend(user, 50, set())
    assert ranks[:50].tolist() == list(range(1, 51))
    assert np.all(np.diff(scores[:50]) <= 0)
    assert np.all(ranks[50:] == 51)        # popular-only candidates


def test_item_features_match_the_history(setup):
    train, _, _, builder = setup
    item = int(builder.item_ids[0])
    rows = train[train["item_id"] == item]
    likes = (rows["rating"] >= 4.0).sum()
    assert builder.item_features["item_likes_all"][0] == pytest.approx(np.log1p(likes))
    assert builder.item_features["item_n_ratings"][0] == pytest.approx(np.log1p(len(rows)))
    assert builder.item_features["item_mean_rating"][0] == pytest.approx(rows["rating"].mean())
    assert builder.item_features["item_year"][0] == 1990 + item % 30


def test_user_without_embedding_gets_popular_candidates_with_missing_als_features(setup):
    _, _, _, builder = setup
    items, features = builder.candidates(10_000, set())
    assert len(items) == 20
    assert np.isnan(features[:, FEATURES.index("als_score")]).all()
    assert np.isnan(features[:, FEATURES.index("genre_match")]).all()
    # The first popular candidate is the most liked item of the window.
    assert features[0, FEATURES.index("item_pop_rank")] == 1


def test_training_labels_mark_the_liked_candidates(setup):
    train, val, _, builder = setup
    groups = build_training_set(builder, train, val)
    assert groups
    liked = val[val["rating"] >= 4.0].groupby("user_id")["item_id"].agg(set)
    for features, labels in groups:
        assert len(features) == len(labels)
        assert 1 <= labels.sum() <= max(len(items) for items in liked)


def test_trained_ranker_recommends_k_unseen_items_and_is_deterministic(setup):
    train, val, _, builder = setup
    groups = build_training_set(builder, train, val)
    first = train_ranker(groups, seed=0, threads=1, min_data_in_leaf=5)
    second = train_ranker(groups, seed=0, threads=1, min_data_in_leaf=5)
    ranked = RankedRecommender(builder, first)

    seen = train.groupby("user_id")["item_id"].agg(set)
    user = next(iter(builder.als.user_row))
    recommended = ranked.recommend(user, 10, seen[user])
    assert len(recommended) == len(set(recommended)) == 10
    assert not seen[user] & set(recommended)
    assert recommended == RankedRecommender(builder, second).recommend(user, 10, seen[user])
    # Re-ranking only reorders: the items come from the same candidate list.
    assert set(recommended) <= set(CandidateRecommender(builder).recommend(user, 70, seen[user]))
    assert evaluate(ranked, train, val)["n_users_evaluated"] > 0
