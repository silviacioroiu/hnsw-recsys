import pandas as pd

from recsys.popularity import SECONDS_PER_DAY, PopularityRecommender


def _interactions(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_ranks_by_number_of_positive_ratings():
    data = _interactions([
        (1, 10, 5.0, 1), (2, 10, 4.0, 2), (3, 10, 4.5, 3),
        (1, 20, 5.0, 4), (2, 20, 4.0, 5),
        (1, 30, 5.0, 6),
    ])
    model = PopularityRecommender().fit(data)
    assert model.recommend(99, 3, set()) == [10, 20, 30]


def test_low_ratings_do_not_count():
    data = _interactions([
        (1, 10, 2.0, 1), (2, 10, 3.5, 2), (3, 10, 1.0, 3),
        (1, 20, 4.0, 4),
    ])
    assert PopularityRecommender().fit(data).recommend(99, 5, set()) == [20]


def test_ties_are_broken_by_item_id():
    data = _interactions([(1, 30, 5.0, 1), (1, 10, 5.0, 2), (1, 20, 5.0, 3)])
    assert PopularityRecommender().fit(data).recommend(99, 3, set()) == [10, 20, 30]


def test_excluded_items_are_skipped_and_k_is_still_filled():
    data = _interactions([
        (1, 10, 5.0, 1), (2, 10, 5.0, 2),
        (1, 20, 5.0, 3), (1, 30, 5.0, 4),
    ])
    model = PopularityRecommender().fit(data)
    assert model.recommend(1, 2, exclude={10}) == [20, 30]


def test_window_counts_only_recent_interactions():
    day = SECONDS_PER_DAY
    data = _interactions([
        (1, 10, 5.0, 0), (2, 10, 5.0, 1), (3, 10, 5.0, 2),   # old favourite
        (1, 20, 5.0, 100 * day), (2, 20, 5.0, 101 * day),     # recent
    ])
    assert PopularityRecommender().fit(data).recommend(99, 1, set()) == [10]
    assert PopularityRecommender(window_days=30).fit(data).recommend(99, 5, set()) == [20]
