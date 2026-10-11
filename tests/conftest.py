import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ratings() -> pd.DataFrame:
    """Small synthetic ratings table in the shape load_ratings() returns."""
    rng = np.random.default_rng(0)
    n = 5_000
    df = pd.DataFrame({
        "user_id": rng.integers(1, 200, n).astype("int32"),
        "item_id": rng.integers(1, 300, n).astype("int32"),
        "rating": rng.choice(np.arange(0.5, 5.5, 0.5), n).astype("float32"),
        "timestamp": rng.integers(1_000_000_000, 1_100_000_000, n),
    })
    # MovieLens has at most one rating per (user, item).
    return df.drop_duplicates(["user_id", "item_id"]).reset_index(drop=True)
