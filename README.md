# hnsw-recsys

Two-stage movie recommender on MovieLens 25M with a from-scratch HNSW index in C++.
Work in progress; the full write-up comes with the last milestone.

## Reproduce (Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
make setup
make test
make m1      # downloads ml-25m, splits by time, writes results/m1_popularity.json
```
