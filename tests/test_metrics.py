import math

import pytest

from recsys.metrics import ndcg_at_k, recall_at_k


def test_worked_example():
    # 3 relevant items, hits at ranks 1 and 4 of the top 10.
    recommended = [10, 1, 2, 11, 3, 4, 5, 6, 7, 8]
    relevant = {10, 11, 12}
    assert recall_at_k(recommended, relevant, 10) == pytest.approx(2 / 3)
    dcg = 1 / math.log2(2) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(recommended, relevant, 10) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(recommended, relevant, 10) == pytest.approx(0.6713, abs=1e-4)


def test_no_hits():
    assert recall_at_k([1, 2, 3], {9}, 3) == 0.0
    assert ndcg_at_k([1, 2, 3], {9}, 3) == 0.0


def test_perfect_list_with_fewer_relevant_than_k():
    # One relevant item at rank 1: the ideal list is also one hit, so NDCG = 1.
    assert recall_at_k([9, 1, 2], {9}, 3) == 1.0
    assert ndcg_at_k([9, 1, 2], {9}, 3) == 1.0


def test_more_relevant_than_k_caps_recall_not_ndcg():
    recommended = [1, 2]
    relevant = {1, 2, 3, 4}
    assert recall_at_k(recommended, relevant, 2) == 0.5
    assert ndcg_at_k(recommended, relevant, 2) == 1.0


def test_hits_beyond_k_are_ignored():
    recommended = list(range(10)) + [99]
    assert recall_at_k(recommended, {99}, 10) == 0.0
    assert recall_at_k(recommended, {99}, 50) == 1.0


def test_lower_rank_scores_lower_ndcg():
    assert ndcg_at_k([9, 1, 2], {9}, 3) > ndcg_at_k([1, 2, 9], {9}, 3)


def test_empty_relevant_is_an_error():
    with pytest.raises(ValueError):
        recall_at_k([1], set(), 1)
    with pytest.raises(ValueError):
        ndcg_at_k([1], set(), 1)
