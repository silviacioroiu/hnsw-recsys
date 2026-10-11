"""Download MovieLens 25M and load its ratings.

Usage: python -m recsys.data [--out data]
"""

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASET = "ml-25m"
URL = f"https://files.grouplens.org/datasets/movielens/{DATASET}.zip"
MD5_URL = URL + ".md5"
# Only these members are extracted; the zip also holds tags and genome scores.
MEMBERS = ["ratings.csv", "movies.csv"]


def file_digests(path: Path) -> dict:
    """MD5 and SHA-256 of a file, read in chunks so the zip never sits in RAM."""
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def download(out_dir: Path) -> Path:
    """Download and extract the dataset; returns the directory with the CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{DATASET}.zip"
    if not zip_path.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)

    digests = file_digests(zip_path)
    with urllib.request.urlopen(MD5_URL) as response:
        published = re.search(r"\b[0-9a-f]{32}\b", response.read().decode())
    if published is None or published.group() != digests["md5"]:
        raise RuntimeError(f"MD5 mismatch for {zip_path}; delete it and retry")

    with zipfile.ZipFile(zip_path) as archive:
        for name in MEMBERS:
            archive.extract(f"{DATASET}/{name}", out_dir)

    # Recorded so every result file can name the exact dataset it came from.
    info = {"name": DATASET, "url": URL, **digests}
    (out_dir / DATASET / "dataset.json").write_text(json.dumps(info, indent=2))
    return out_dir / DATASET


def load_ratings(csv_path: Path) -> pd.DataFrame:
    """Load ratings.csv as (user_id, item_id, rating, timestamp)."""
    ratings = pd.read_csv(
        csv_path,
        dtype={"userId": "int32", "movieId": "int32",
               "rating": "float32", "timestamp": "int64"},
    )
    return ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(f"Dataset ready in {download(args.out)}")


if __name__ == "__main__":
    main()
