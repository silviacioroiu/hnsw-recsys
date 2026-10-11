"""Per-user ranking metrics with binary relevance."""

import math
from collections.abc import Sequence


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of the user's relevant items that appear in the top k.

    The denominator is |relevant|, not min(k, |relevant|): a user with 30
    relevant items cannot reach Recall@10 = 1. Papers differ on this.
    """
    if not relevant:
        raise ValueError("recall is undefined for a user with no relevant items")
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """DCG of the top k divided by the DCG of a perfect top k."""
    if not relevant:
        raise ValueError("NDCG is undefined for a user with no relevant items")
    # Rank r (1-based) contributes 1 / log2(r + 1); enumerate is 0-based.
    dcg = sum(1.0 / math.log2(i + 2)
              for i, item in enumerate(recommended[:k]) if item in relevant)
    # A perfect list puts min(k, |relevant|) hits at the top.
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal
