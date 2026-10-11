.PHONY: setup bindings data split m1 m2 m4 m5 test cpp-test

setup:
	python -m pip install -e ".[dev,bench]"

# Compiles the C++ index into src/recsys/_hnsw*.so
bindings:
	cmake -S cpp -B build/cpp -DHNSW_BUILD_PYTHON=ON -Dpybind11_DIR=$$(python -m pybind11 --cmakedir)
	cmake --build build/cpp -j --target _hnsw

# Downloads ml-25m.zip (~250 MB), checks it against the published MD5.
data:
	python -m recsys.data

split:
	python -m recsys.split

# Reproduces the M1 table: results/m1_popularity.json
m1: data split
	python -m recsys.popularity

# Reproduces the M2 results: results/m2_als.json (needs the split from m1)
m2:
	python -m recsys.als

# Reproduces the M4 results: results/m4_retrieval.json, m4_ann_benchmark.json
# and the recall-vs-latency plot (needs the embeddings from m2)
m4: bindings
	python -m recsys.ann
	python benchmarks/ann_benchmark.py

# Reproduces the M5 results: results/m5_ranking.json (needs m2)
m5:
	python -m recsys.ranking

test:
	python -m pytest -q

# Builds the C++ index and runs its GoogleTest suite.
cpp-test:
	cmake -S cpp -B build/cpp
	cmake --build build/cpp -j
	./build/cpp/hnsw_tests --gtest_brief=1
