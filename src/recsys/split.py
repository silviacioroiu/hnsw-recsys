"""Global time-based train / validation / test split.

Usage: python -m recsys.split [--raw data/ml-25m] [--out data/processed]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import load_ratings


def temporal_split(ratings: pd.DataFrame, val_q: float = 0.8, test_q: float = 0.9):
    """Split by two global timestamp cutoffs.

    train: t < t_val     val: t_val <= t < t_test     test: t >= t_test

    Returns (train, val, test, cutoffs).
    """
    timestamps = ratings["timestamp"].to_numpy()
    # method="lower" returns a timestamp that exists in the data, so the
    # cutoffs are exact integers and the split is reproducible bit for bit.
    t_val, t_test = (int(t) for t in np.quantile(timestamps, [val_q, test_q], method="lower"))
    if not timestamps.min() < t_val < t_test:
        raise ValueError("Cutoffs are degenerate: too few distinct timestamps")

    train = ratings[timestamps < t_val]
    val = ratings[(timestamps >= t_val) & (timestamps < t_test)]
    test = ratings[timestamps >= t_test]
    return train, val, test, {"t_val": t_val, "t_test": t_test}


def _iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/ml-25m"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-q", type=float, default=0.8)
    parser.add_argument("--test-q", type=float, default=0.9)
    args = parser.parse_args()

    ratings = load_ratings(args.raw / "ratings.csv")
    train, val, test, cutoffs = temporal_split(ratings, args.val_q, args.test_q)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(args.out / f"{name}.parquet", index=False)

    dataset_file = args.raw / "dataset.json"
    info = {
        "dataset": json.loads(dataset_file.read_text()) if dataset_file.exists() else None,
        "val_q": args.val_q,
        "test_q": args.test_q,
        **cutoffs,
        "t_val_utc": _iso(cutoffs["t_val"]),
        "t_test_utc": _iso(cutoffs["t_test"]),
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
    }
    (args.out / "split.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
