// Hierarchical Navigable Small World index.
// Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
// using Hierarchical Navigable Small World graphs" (2018), arXiv:1603.09320.
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace hnsw {

enum class Metric {
    L2,            // squared Euclidean distance
    InnerProduct,  // negated dot product, so that smaller is still closer
};

struct Neighbor {
    float distance;
    uint32_t id;

    // Closest first; ties broken by id so results are deterministic.
    bool operator<(const Neighbor& other) const {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }
    bool operator>(const Neighbor& other) const { return other < *this; }
};

class Index {
public:
    // M: links per node on layers >= 1 (layer 0 allows 2 * M).
    // ef_construction: size of the candidate list while inserting.
    // heuristic: choose neighbours with the paper's Algorithm 4 (diverse
    // directions) instead of Algorithm 3 (simply the closest).
    Index(size_t dim, Metric metric, size_t M = 16, size_t ef_construction = 200,
          uint64_t seed = 0, bool heuristic = false);

    // Inserts a copy of `vector` (dim floats). Returns its id: 0, 1, 2, ...
    // Paper: Algorithm 1 (INSERT).
    uint32_t add(const float* vector);

    // The k nearest stored vectors to `query`, closest first. `ef` is the size
    // of the candidate list on layer 0; values below k are raised to k.
    // Paper: Algorithm 5 (K-NN-SEARCH).
    std::vector<Neighbor> search(const float* query, size_t k, size_t ef) const;

    size_t size() const { return links_.size(); }
    size_t dim() const { return dim_; }
    // Highest layer in the graph; -1 while the index is empty.
    int max_level() const { return max_level_; }
    // Highest layer that node `id` belongs to.
    int level(uint32_t id) const { return static_cast<int>(links_[id].size()) - 1; }
    // Maximum number of links a node may keep on `layer`.
    size_t max_degree(int layer) const { return layer == 0 ? 2 * M_ : M_; }

    const std::vector<uint32_t>& neighbors(uint32_t id, int layer) const {
        return links_[id][layer];
    }
    const float* vector(uint32_t id) const { return &data_[id * dim_]; }
    float distance(const float* a, const float* b) const;

private:
    // Greedy best-first search on one layer, starting from `entry_points`.
    // Returns up to `ef` nearest nodes found, closest first.
    // Paper: Algorithm 2 (SEARCH-LAYER).
    std::vector<Neighbor> search_layer(const float* query,
                                       const std::vector<uint32_t>& entry_points,
                                       size_t ef, int layer) const;

    // Up to `m` neighbours for a node, chosen from `candidates` (whose
    // distances are measured from that node).
    // Paper: Algorithm 3 (SELECT-NEIGHBORS-SIMPLE), or Algorithm 4
    // (SELECT-NEIGHBORS-HEURISTIC) when the index was built with heuristic.
    std::vector<uint32_t> select_neighbors(std::vector<Neighbor> candidates, size_t m) const;

    // Draws a node's top layer: floor(-ln(uniform(0,1]) * 1/ln(M)).
    int random_level();

    // Copies the vector into the index with empty link lists for layers
    // 0..level, and returns the new id. Does not connect it to anything.
    uint32_t store(const float* vector, int level);

    size_t dim_;
    Metric metric_;
    size_t M_;
    size_t ef_construction_;
    bool heuristic_;
    double level_mult_;
    std::mt19937_64 rng_;

    std::vector<float> data_;                               // n * dim, row-major
    std::vector<std::vector<std::vector<uint32_t>>> links_;  // links_[id][layer]
    uint32_t entry_point_ = 0;
    int max_level_ = -1;
};

}  // namespace hnsw
