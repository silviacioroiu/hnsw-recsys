import numpy as np
import pandas as pd
import pytest

pytest.importorskip("recsys._hnsw")   # built by `make bindings`

from fastapi.testclient import TestClient  # noqa: E402

from recsys.als import ALSRecommender  # noqa: E402
from recsys.features import FeatureBuilder  # noqa: E402
from recsys.ranking import RankedRecommender, build_training_set, train_ranker  # noqa: E402
from recsys.service import create_app  # noqa: E402
from recsys.serving import ServingPipeline  # noqa: E402
from recsys.split import temporal_split  # noqa: E402


@pytest.fixture
def movies() -> pd.DataFrame:
    return pd.DataFrame({
        "item_id": np.arange(1, 300),
        "title": [f"Film {i} (2000)" for i in range(1, 300)],
        "genres": ["Drama" if i % 2 else "Comedy|Action" for i in range(1, 300)],
        "year": np.full(299, 2000, dtype="float32"),
    })


@pytest.fixture
def served(ratings, movies, tmp_path):
    """Trains a tiny system, exports it, and loads it back from disk."""
    train, val, _, _ = temporal_split(ratings)
    als = ALSRecommender(factors=8, iterations=5, seed=0, num_threads=1).fit(train)
    als.save(tmp_path / "als")
    builder = FeatureBuilder(train, movies, als, n_als=50, n_popular=20)
    model = train_ranker(build_training_set(builder, train, val), seed=0, threads=1,
                         min_data_in_leaf=5)
    (tmp_path / "ranker").mkdir()
    model.save_model(tmp_path / "ranker" / "model.txt")
    ServingPipeline.export(tmp_path, train, movies, builder)
    return train, builder, model, ServingPipeline.load(tmp_path)


def test_served_pipeline_matches_the_offline_ranker(served):
    # 50 ALS candidates out of ~300 items: the index search is wide enough to
    # be exact here, so the reloaded pipeline must reproduce the offline one.
    train, builder, model, pipeline = served
    offline = RankedRecommender(builder, model)
    seen = train.groupby("user_id")["item_id"].agg(set)
    for user in list(builder.als.user_row)[:30]:
        assert pipeline.recommend(user, 10) == offline.recommend(user, 10, seen[user])


def test_recommend_endpoint_returns_unseen_items_with_titles(served):
    train, _, _, pipeline = served
    client = TestClient(create_app(pipeline))
    user = int(train["user_id"].iloc[0])

    response = client.get(f"/recommend/{user}?k=5")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == user and body["known_user"] is True
    ids = [item["item_id"] for item in body["items"]]
    assert len(ids) == len(set(ids)) == 5
    assert not set(ids) & set(train.loc[train["user_id"] == user, "item_id"])
    assert all(item["title"] == f"Film {item['item_id']} (2000)" for item in body["items"])


def test_k_defaults_to_ten_and_is_validated(served):
    train, _, _, pipeline = served
    client = TestClient(create_app(pipeline))
    user = int(train["user_id"].iloc[0])
    assert len(client.get(f"/recommend/{user}").json()["items"]) == 10
    assert client.get(f"/recommend/{user}?k=0").status_code == 422
    assert client.get(f"/recommend/{user}?k=101").status_code == 422
    assert client.get("/recommend/not-a-number").status_code == 422


def test_unknown_user_gets_popular_items(served):
    _, _, _, pipeline = served
    body = TestClient(create_app(pipeline)).get("/recommend/999999?k=5").json()
    assert body["known_user"] is False
    assert len(body["items"]) == 5


def test_health(served):
    _, builder, _, pipeline = served
    body = TestClient(create_app(pipeline)).get("/health").json()
    assert body == {"status": "ok", "items": len(builder.item_ids),
                    "users": len(pipeline.seen_users)}
