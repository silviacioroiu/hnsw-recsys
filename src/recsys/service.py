"""HTTP API for the recommender.

Run: RECSYS_ARTIFACTS=data/artifacts uvicorn recsys.service:app_from_env --factory
"""

import os
from pathlib import Path

from fastapi import FastAPI, Query

from recsys.serving import ServingPipeline


def create_app(pipeline: ServingPipeline) -> FastAPI:
    app = FastAPI(title="hnsw-recsys")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "items": len(pipeline.titles), "users": len(pipeline.seen_users)}

    # A plain `def` endpoint runs in FastAPI's thread pool, so one slow
    # request does not block the event loop for the others.
    @app.get("/recommend/{user_id}")
    def recommend(user_id: int, k: int = Query(10, ge=1, le=100)) -> dict:
        rows = pipeline.recommend_rows(user_id, k)
        item_ids = pipeline.builder.item_ids[rows].tolist()
        return {
            "user_id": user_id,
            # Unknown users still get an answer: the popular candidates, ranked.
            "known_user": pipeline.knows(user_id),
            "items": [{"item_id": item_id, "title": str(pipeline.titles[row])}
                      for item_id, row in zip(item_ids, rows.tolist())],
        }

    return app


def app_from_env() -> FastAPI:
    """Factory for uvicorn; loads the pipeline once, before serving starts."""
    return create_app(ServingPipeline.load(Path(os.environ.get("RECSYS_ARTIFACTS",
                                                               "data/artifacts"))))
