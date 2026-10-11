import numpy as np
import pytest

# Built by `make bindings`; skipped where the extension has not been compiled.
_hnsw = pytest.importorskip("recsys._hnsw")


@pytest.fixture
def vectors() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((2000, 16)).astype(np.float32)


def test_matches_exact_inner_product_search(vectors):
    index = _hnsw.Index(dim=16, metric="ip", M=16, ef_construction=200, seed=0)
    index.add(vectors)
    assert len(index) == 2000

    queries = np.random.default_rng(1).standard_normal((100, 16)).astype(np.float32)
    hits = 0
    for query in queries:
        ids, distances = index.search(query, k=10, ef=200)
        assert ids.dtype == np.uint32 and len(ids) == 10
        # Distances are negated dot products, closest (largest dot) first.
        np.testing.assert_allclose(distances, -(vectors[ids] @ query), rtol=1e-4)
        assert np.all(np.diff(distances) >= 0)
        exact = np.argsort(-(vectors @ query))[:10]
        hits += len(set(ids.tolist()) & set(exact.tolist()))
    assert hits / 1000 >= 0.90


def test_l2_finds_the_vector_itself(vectors):
    index = _hnsw.Index(dim=16, metric="l2")
    index.add(vectors)
    ids, distances = index.search(vectors[123], k=1, ef=50)
    assert ids.tolist() == [123]
    assert distances[0] == pytest.approx(0.0, abs=1e-5)


def test_accepts_float64_and_non_contiguous_input(vectors):
    index = _hnsw.Index(dim=8, metric="l2")
    index.add(vectors.astype(np.float64)[:, ::2])   # converted to float32 copies
    ids, _ = index.search(vectors[7, ::2], k=1, ef=50)
    assert ids.tolist() == [7]


def test_rejects_wrong_shapes_and_metric(vectors):
    index = _hnsw.Index(dim=16)
    with pytest.raises(ValueError):
        index.add(vectors[:, :8])
    with pytest.raises(ValueError):
        index.search(vectors[0, :8], k=1, ef=10)
    with pytest.raises(ValueError):
        _hnsw.Index(dim=16, metric="cosine")


def test_recommender_agrees_with_exact_search_and_filters_seen_items():
    from recsys.als import ALSRecommender
    from recsys.ann import HNSWRecommender

    rng = np.random.default_rng(0)
    als = ALSRecommender()
    als.user_factors = rng.standard_normal((20, 8)).astype(np.float32)
    als.item_factors = rng.standard_normal((300, 8)).astype(np.float32)
    als.user_ids = np.arange(20)
    als.item_ids = np.arange(1000, 1300)          # raw ids differ from row numbers
    als.user_row = {user: user for user in range(20)}

    hnsw = HNSWRecommender(als, ef=300)            # ef = n: the search is exhaustive
    for user in range(20):
        exclude = set(als.recommend(user, 3, set()))   # pretend the top 3 were seen
        recommended = hnsw.recommend(user, 10, exclude)
        assert recommended == als.recommend(user, 10, exclude)
        assert not exclude & set(recommended)
    assert hnsw.recommend(999, 5, set()) == []     # unknown user, no fallback set


def test_empty_index_returns_empty_arrays():
    ids, distances = _hnsw.Index(dim=4).search(np.zeros(4, dtype=np.float32), k=5, ef=10)
    assert len(ids) == 0 and len(distances) == 0
