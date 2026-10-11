#include "hnsw/hnsw.h"

#include <algorithm>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

using hnsw::Index;
using hnsw::Metric;
using hnsw::Neighbor;

namespace {

const size_t kDim = 16;

std::vector<float> random_vectors(size_t n, size_t dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (float& x : data) x = normal(rng);
    return data;
}

Index build(const std::vector<float>& data, size_t dim, Metric metric, size_t M = 16,
            size_t ef_construction = 200, uint64_t seed = 0, bool heuristic = false) {
    Index index(dim, metric, M, ef_construction, seed, heuristic);
    for (size_t i = 0; i < data.size() / dim; ++i) index.add(&data[i * dim]);
    return index;
}

// Exact answer: compare the query with every stored vector.
std::vector<Neighbor> brute_force(const Index& index, const float* query, size_t k) {
    std::vector<Neighbor> all;
    for (uint32_t id = 0; id < index.size(); ++id) {
        all.push_back({index.distance(query, index.vector(id)), id});
    }
    std::sort(all.begin(), all.end());
    if (all.size() > k) all.resize(k);
    return all;
}

// Mean fraction of the exact top k that the index returns.
double recall(const Index& index, const std::vector<float>& queries, size_t k, size_t ef) {
    const size_t n_queries = queries.size() / index.dim();
    size_t found = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        const float* query = &queries[q * index.dim()];
        std::set<uint32_t> truth;
        for (const Neighbor& n : brute_force(index, query, k)) truth.insert(n.id);
        for (const Neighbor& n : index.search(query, k, ef)) found += truth.count(n.id);
    }
    return static_cast<double>(found) / static_cast<double>(n_queries * k);
}

}  // namespace

// --- Parameters and distances. ----------------------------------------------

TEST(Basics,RejectsInvalidParameters) {
    EXPECT_THROW(Index(0, Metric::L2), std::invalid_argument);
    EXPECT_THROW(Index(4, Metric::L2, 1), std::invalid_argument);
    EXPECT_THROW(Index(4, Metric::L2, 16, 0), std::invalid_argument);
}

TEST(Basics,SquaredL2Distance) {
    Index index(3, Metric::L2);
    const float a[] = {1, 2, 3}, b[] = {4, 6, 3};
    EXPECT_FLOAT_EQ(index.distance(a, b), 25.0f);
    EXPECT_FLOAT_EQ(index.distance(a, a), 0.0f);
}

TEST(Basics,InnerProductIsNegatedSoSmallerIsCloser) {
    Index index(3, Metric::InnerProduct);
    const float a[] = {1, 2, 3}, b[] = {4, 5, 6};
    EXPECT_FLOAT_EQ(index.distance(a, b), -32.0f);
}

TEST(Basics,LayerZeroAllowsTwiceTheLinks) {
    Index index(4, Metric::L2, 8);
    EXPECT_EQ(index.max_degree(0), 16u);
    EXPECT_EQ(index.max_degree(1), 8u);
    EXPECT_EQ(index.max_level(), -1);
}

// --- Insertion, search and graph structure. ---------------------------------

TEST(Search, EmptyIndexReturnsNothing) {
    Index index(kDim, Metric::L2);
    const std::vector<float> query(kDim, 0.0f);
    EXPECT_TRUE(index.search(query.data(), 10, 50).empty());
}

TEST(Add, ReturnsSequentialIdsAndStoresACopy) {
    Index index(2, Metric::L2);
    float v[] = {1.0f, 2.0f};
    EXPECT_EQ(index.add(v), 0u);
    v[0] = 9.0f;  // the index must not keep a pointer to the caller's memory
    EXPECT_EQ(index.add(v), 1u);
    EXPECT_EQ(index.size(), 2u);
    EXPECT_FLOAT_EQ(index.vector(0)[0], 1.0f);
    EXPECT_FLOAT_EQ(index.vector(1)[0], 9.0f);
}

TEST(Search, SingleElement) {
    Index index(2, Metric::L2);
    const float v[] = {1.0f, 2.0f};
    index.add(v);
    const auto result = index.search(v, 5, 10);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].id, 0u);
    EXPECT_FLOAT_EQ(result[0].distance, 0.0f);
}

TEST(Search, ReturnsEverythingWhenKExceedsSize) {
    const auto data = random_vectors(5, kDim, 1);
    const Index index = build(data, kDim, Metric::L2);
    EXPECT_EQ(index.search(data.data(), 10, 10).size(), 5u);
}

TEST(Search, ResultsAreSortedUniqueAndExactlyK) {
    const auto data = random_vectors(500, kDim, 2);
    const auto queries = random_vectors(20, kDim, 3);
    const Index index = build(data, kDim, Metric::L2);
    for (size_t q = 0; q < 20; ++q) {
        const float* query = &queries[q * kDim];
        const auto result = index.search(query, 10, 50);
        ASSERT_EQ(result.size(), 10u);
        std::set<uint32_t> ids;
        for (size_t i = 0; i < result.size(); ++i) {
            ids.insert(result[i].id);
            EXPECT_FLOAT_EQ(result[i].distance, index.distance(query, index.vector(result[i].id)));
            if (i > 0) {
                EXPECT_LE(result[i - 1].distance, result[i].distance);
            }
        }
        EXPECT_EQ(ids.size(), 10u);
    }
}

TEST(Search, EfBelowKStillReturnsK) {
    const auto data = random_vectors(200, kDim, 4);
    const Index index = build(data, kDim, Metric::L2);
    EXPECT_EQ(index.search(data.data(), 10, 1).size(), 10u);
}

TEST(Search, FindsAStoredVectorAsItsOwnNearestNeighbour) {
    const auto data = random_vectors(1000, kDim, 5);
    const Index index = build(data, kDim, Metric::L2);
    size_t hits = 0;
    for (uint32_t id = 0; id < 1000; ++id) {
        const auto result = index.search(&data[id * kDim], 1, 50);
        ASSERT_EQ(result.size(), 1u);
        hits += result[0].id == id;
    }
    EXPECT_GE(hits, 990u);
}

TEST(Recall, L2MatchesBruteForce) {
    const auto data = random_vectors(2000, kDim, 6);
    const auto queries = random_vectors(100, kDim, 7);
    const Index index = build(data, kDim, Metric::L2);
    EXPECT_GE(recall(index, queries, 10, 100), 0.95);
}

TEST(Recall, InnerProductMatchesBruteForce) {
    const auto data = random_vectors(2000, kDim, 8);
    const auto queries = random_vectors(100, kDim, 9);
    const Index index = build(data, kDim, Metric::InnerProduct);
    EXPECT_GE(recall(index, queries, 10, 100), 0.90);
}

TEST(Recall, GrowsWithEf) {
    const auto data = random_vectors(2000, kDim, 10);
    const auto queries = random_vectors(100, kDim, 11);
    // A sparse graph (small M) so that a small ef visibly loses recall.
    const Index index = build(data, kDim, Metric::L2, 4, 40);
    const double low = recall(index, queries, 10, 10);
    const double high = recall(index, queries, 10, 200);
    EXPECT_LT(low, high);
    EXPECT_GE(high, 0.90);
}

TEST(Graph, LinksRespectDegreeLimitsAndLayers) {
    const auto data = random_vectors(2000, kDim, 12);
    const Index index = build(data, kDim, Metric::L2, 8, 100);
    for (uint32_t id = 0; id < index.size(); ++id) {
        for (int layer = 0; layer <= index.level(id); ++layer) {
            const auto& links = index.neighbors(id, layer);
            EXPECT_LE(links.size(), index.max_degree(layer));
            const std::set<uint32_t> unique(links.begin(), links.end());
            EXPECT_EQ(unique.size(), links.size()) << "duplicate link at node " << id;
            for (uint32_t other : links) {
                ASSERT_LT(other, index.size());
                EXPECT_NE(other, id) << "self link at node " << id;
                EXPECT_GE(index.level(other), layer) << "link to a node missing from the layer";
            }
        }
    }
}

TEST(Graph, EveryNodeButTheFirstHasALinkOnLayerZero) {
    const auto data = random_vectors(2000, kDim, 13);
    const Index index = build(data, kDim, Metric::L2, 8, 100);
    for (uint32_t id = 0; id < index.size(); ++id) {
        EXPECT_FALSE(index.neighbors(id, 0).empty()) << "isolated node " << id;
    }
}

TEST(Graph, LevelsFollowTheExponentialDistribution) {
    const size_t n = 5000, M = 16;
    const auto data = random_vectors(n, 4, 14);
    const Index index = build(data, 4, Metric::L2, M, 20);
    size_t above_zero = 0;
    int highest = 0;
    for (uint32_t id = 0; id < n; ++id) {
        above_zero += index.level(id) >= 1;
        highest = std::max(highest, index.level(id));
    }
    // P(level >= 1) = 1 / M = 0.0625
    const double fraction = static_cast<double>(above_zero) / n;
    EXPECT_GT(fraction, 0.045);
    EXPECT_LT(fraction, 0.080);
    EXPECT_EQ(index.max_level(), highest);
}

// --- Heuristic neighbour selection (Algorithm 4). ---------------------------

TEST(Heuristic, KeepsRecallAndDegreeLimits) {
    const auto data = random_vectors(2000, kDim, 16);
    const auto queries = random_vectors(100, kDim, 17);
    const Index index = build(data, kDim, Metric::L2, 16, 200, 0, true);
    EXPECT_GE(recall(index, queries, 10, 100), 0.95);
    for (uint32_t id = 0; id < index.size(); ++id) {
        EXPECT_FALSE(index.neighbors(id, 0).empty()) << "isolated node " << id;
        for (int layer = 0; layer <= index.level(id); ++layer) {
            EXPECT_LE(index.neighbors(id, layer).size(), index.max_degree(layer));
        }
    }
}

TEST(Heuristic, NoKeptNeighbourIsCloserToAnotherThanToTheNode) {
    // Checked on the last node inserted: nothing has pruned its links since.
    const auto data = random_vectors(500, kDim, 18);
    const Index index = build(data, kDim, Metric::L2, 8, 100, 0, true);
    const uint32_t last = static_cast<uint32_t>(index.size()) - 1;
    const auto& links = index.neighbors(last, 0);
    ASSERT_FALSE(links.empty());
    for (size_t i = 0; i < links.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            const float to_node = index.distance(index.vector(links[i]), index.vector(last));
            const float to_kept = index.distance(index.vector(links[i]), index.vector(links[j]));
            EXPECT_GE(to_kept, to_node);
        }
    }
}

TEST(Heuristic, BuildsASparserGraphThanSimpleSelection) {
    const auto data = random_vectors(2000, kDim, 19);
    size_t links[2] = {0, 0};
    for (int heuristic = 0; heuristic < 2; ++heuristic) {
        const Index index = build(data, kDim, Metric::L2, 16, 200, 0, heuristic == 1);
        for (uint32_t id = 0; id < index.size(); ++id) {
            links[heuristic] += index.neighbors(id, 0).size();
        }
    }
    EXPECT_LT(links[1], links[0]);
}

TEST(Graph, SameSeedBuildsTheSameGraph) {
    const auto data = random_vectors(500, kDim, 15);
    const Index first = build(data, kDim, Metric::L2, 8, 50, 42);
    const Index second = build(data, kDim, Metric::L2, 8, 50, 42);
    for (uint32_t id = 0; id < first.size(); ++id) {
        ASSERT_EQ(first.level(id), second.level(id));
        for (int layer = 0; layer <= first.level(id); ++layer) {
            EXPECT_EQ(first.neighbors(id, layer), second.neighbors(id, layer));
        }
    }
}
