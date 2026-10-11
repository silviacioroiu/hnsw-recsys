"""Candidate generation and features for the re-ranking stage.

Everything here is computed from `history` only, so a builder made from the
train window can never see the validation labels it is used to learn from.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recsys.als import ALSRecommender

SECONDS_PER_DAY = 86_400

FEATURES = [
    # user x item
    "als_score", "als_rank", "genre_match",
    # item
    "item_likes_all", "item_likes_30d", "item_likes_7d", "item_pop_rank",
    "item_mean_rating", "item_n_ratings", "item_age_days", "item_year",
    # user
    "user_n_ratings", "user_n_likes", "user_mean_rating", "user_idle_days",
]


class FeatureBuilder:
    """Candidates for a user and one feature row per candidate.

    Candidates are the ALS top `n_als` plus the `n_popular` most liked items of
    the last `window_days`, since recent popularity finds items ALS misses.
    Only items with an ALS embedding (at least one like in the history) can be
    candidates; item arrays are aligned with `als.item_ids`.
    """

    def __init__(self, history: pd.DataFrame, movies: pd.DataFrame, als: ALSRecommender,
                 min_rating: float = 4.0, n_als: int = 500, n_popular: int = 100,
                 window_days: int = 30):
        self.als = als
        self.n_als = n_als
        self.n_popular = n_popular
        self.item_ids = als.item_ids
        self.item_row = {int(item): row for row, item in enumerate(self.item_ids)}
        end = int(history["timestamp"].max())
        liked = history[history["rating"] >= min_rating]
        items = pd.Index(self.item_ids)

        def per_item(series: pd.Series, fill: float) -> np.ndarray:
            return series.reindex(items).fillna(fill).to_numpy(dtype=np.float32)

        def likes_since(days: int) -> np.ndarray:
            recent = liked[liked["timestamp"] >= end - days * SECONDS_PER_DAY]
            return per_item(recent.groupby("item_id").size(), 0)

        by_item = history.groupby("item_id")
        likes_window = likes_since(window_days)
        # Rank 1 = most liked in the window; ties broken by item id, exactly
        # like the popularity baseline.
        order = np.lexsort((self.item_ids, -likes_window))
        pop_rank = np.empty(len(items), dtype=np.float32)
        pop_rank[order] = np.arange(1, len(items) + 1)
        self.popular_rows = order[likes_window[order] > 0]

        first_seen = per_item(by_item["timestamp"].min(), end)
        self.item_features = {
            "item_likes_all": np.log1p(per_item(liked.groupby("item_id").size(), 0)),
            "item_likes_30d": np.log1p(likes_window),
            "item_likes_7d": np.log1p(likes_since(7)),
            "item_pop_rank": pop_rank,
            "item_mean_rating": per_item(by_item["rating"].mean(), np.nan),
            "item_n_ratings": np.log1p(per_item(by_item.size(), 0)),
            "item_age_days": (end - first_seen) / SECONDS_PER_DAY,
            "item_year": per_item(movies.set_index("item_id")["year"], np.nan),
        }

        # One 0/1 column per genre, and each item's number of genres.
        genres = (movies.set_index("item_id")["genres"].reindex(items).fillna("")
                  .str.get_dummies("|"))
        self.genres = genres.to_numpy(dtype=np.float32)
        self.genre_count = np.maximum(self.genres.sum(axis=1), 1)

        by_user = history.groupby("user_id")
        stats = pd.DataFrame({
            "n_ratings": by_user.size(),
            "n_likes": liked.groupby("user_id").size(),
            "mean_rating": by_user["rating"].mean(),
            "idle_days": (end - by_user["timestamp"].max()) / SECONDS_PER_DAY,
        }).fillna({"n_likes": 0})
        self.user_stats = dict(zip(stats.index, stats.to_numpy(dtype=np.float32)))

        # Sparse user x item matrix of likes, to read a user's liked items fast.
        user_codes, like_users = pd.factorize(liked["user_id"])
        self.like_row = {int(user): row for row, user in enumerate(like_users)}
        item_codes = items.get_indexer(liked["item_id"])
        self.likes = csr_matrix(
            (np.ones(len(liked), dtype=np.float32), (user_codes, item_codes)),
            shape=(len(like_users), len(items)))

    def _genre_match(self, user_id: int, rows: np.ndarray) -> np.ndarray:
        """How well each item's genres fit the genres of the films the user liked."""
        like_row = self.like_row.get(user_id)
        if like_row is None:
            return np.full(len(rows), np.nan, dtype=np.float32)
        liked_rows = self.likes.indices[self.likes.indptr[like_row]:self.likes.indptr[like_row + 1]]
        profile = self.genres[liked_rows].sum(axis=0)
        profile /= max(profile.sum(), 1.0)
        # Averaged over the item's genres, so many-genre films are not favoured.
        return (self.genres[rows] @ profile) / self.genre_count[rows]

    def candidates(self, user_id: int, exclude: set[int]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (item_ids, features): ALS candidates in score order, then the
        popular ones not already present. features has one column per FEATURES."""
        n_items = len(self.item_ids)
        allowed = np.ones(n_items, dtype=bool)
        allowed[[self.item_row[item] for item in exclude if item in self.item_row]] = False

        user_row = self.als.user_row.get(user_id)
        n = min(self.n_als, int(allowed.sum()))
        if user_row is None or n == 0:
            # No embedding: only popular candidates, with missing ALS features.
            scores = np.full(n_items, np.nan, dtype=np.float32)
            als_rows = np.empty(0, dtype=np.int64)
        else:
            scores = self.als.item_factors @ self.als.user_factors[user_row]
            masked = np.where(allowed, scores, -np.inf)
            als_rows = np.argpartition(-masked, n - 1)[:n]
            als_rows = als_rows[np.argsort(-masked[als_rows], kind="stable")]

        popular = self.popular_rows[allowed[self.popular_rows]][:self.n_popular]
        extra = popular[~np.isin(popular, als_rows)]
        rows = np.concatenate([als_rows, extra])
        # Popular-only candidates all share the rank just past the ALS list.
        als_rank = np.concatenate([np.arange(1, len(als_rows) + 1),
                                   np.full(len(extra), self.n_als + 1)])

        user = self.user_stats.get(user_id, np.full(4, np.nan, dtype=np.float32))
        columns = {
            "als_score": scores[rows],
            "als_rank": als_rank,
            "genre_match": self._genre_match(user_id, rows),
            **{name: values[rows] for name, values in self.item_features.items()},
            "user_n_ratings": np.full(len(rows), np.log1p(user[0])),
            "user_n_likes": np.full(len(rows), np.log1p(user[1])),
            "user_mean_rating": np.full(len(rows), user[2]),
            "user_idle_days": np.full(len(rows), user[3]),
        }
        features = np.column_stack([columns[name] for name in FEATURES]).astype(np.float32)
        return self.item_ids[rows], features
