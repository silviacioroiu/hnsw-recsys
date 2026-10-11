.PHONY: setup data split m1 m2 test cpp-test

setup:
	python -m pip install -e ".[dev]"

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

test:
	python -m pytest -q

# Builds the C++ index and runs its GoogleTest suite.
cpp-test:
	cmake -S cpp -B build/cpp
	cmake --build build/cpp -j
	./build/cpp/hnsw_tests --gtest_brief=1
