import pandas as pd
import pytest

from recsys.split import temporal_split


def test_windows_are_ordered_in_time(ratings):
    train, val, test, cutoffs = temporal_split(ratings)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
    assert val["timestamp"].min() == cutoffs["t_val"]
    assert test["timestamp"].min() == cutoffs["t_test"]


def test_every_row_lands_in_exactly_one_window(ratings):
    train, val, test, _ = temporal_split(ratings)
    assert len(train) + len(val) + len(test) == len(ratings)
    merged = pd.concat([train, val, test]).sort_index()
    pd.testing.assert_frame_equal(merged, ratings)


def test_window_sizes_follow_the_quantiles(ratings):
    train, val, test, _ = temporal_split(ratings, 0.8, 0.9)
    n = len(ratings)
    assert len(train) / n == pytest.approx(0.8, abs=0.01)
    assert len(val) / n == pytest.approx(0.1, abs=0.01)
    assert len(test) / n == pytest.approx(0.1, abs=0.01)


def test_split_is_deterministic(ratings):
    first = temporal_split(ratings)
    second = temporal_split(ratings.sample(frac=1, random_state=1))
    assert first[3] == second[3]
    for a, b in zip(first[:3], second[:3]):
        pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_identical_timestamps_are_rejected():
    ratings = pd.DataFrame({"user_id": [1, 2, 3], "item_id": [1, 2, 3],
                            "rating": [4.0, 4.0, 4.0], "timestamp": [7, 7, 7]})
    with pytest.raises(ValueError):
        temporal_split(ratings)
