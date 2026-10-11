#include "hnsw/hnsw.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <stdexcept>

namespace hnsw {

Index::Index(size_t dim, Metric metric, size_t M, size_t ef_construction, uint64_t seed,
             bool heuristic)
    : dim_(dim),
      metric_(metric),
      M_(M),
      ef_construction_(ef_construction),
      heuristic_(heuristic),
      level_mult_(1.0 / std::log(static_cast<double>(M))),
      rng_(seed) {
    if (dim == 0) throw std::invalid_argument("dim must be positive");
    if (M < 2) throw std::invalid_argument("M must be at least 2");
    if (ef_construction == 0) throw std::invalid_argument("ef_construction must be positive");
}

float Index::distance(const float* a, const float* b) const {
    float sum = 0.0f;
    if (metric_ == Metric::L2) {
        // Squared distance: the square root would not change the ordering.
        for (size_t i = 0; i < dim_; ++i) {
            const float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
    for (size_t i = 0; i < dim_; ++i) sum += a[i] * b[i];
    return -sum;
}

int Index::random_level() {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double r = 1.0 - uniform(rng_);  // in (0, 1], so log(r) is finite
    return static_cast<int>(-std::log(r) * level_mult_);
}

uint32_t Index::store(const float* vector, int level) {
    const auto id = static_cast<uint32_t>(size());
    data_.insert(data_.end(), vector, vector + dim_);
    links_.emplace_back(level + 1);
    return id;
}

std::vector<uint32_t> Index::select_neighbors(std::vector<Neighbor> candidates, size_t m) const {
    std::sort(candidates.begin(), candidates.end());
    std::vector<uint32_t> ids;
    ids.reserve(std::min(m, candidates.size()));
    for (const Neighbor& candidate : candidates) {
        if (ids.size() == m) break;
        bool keep = true;
        if (heuristic_) {
            // Algorithm 4: skip a candidate that is closer to a neighbour we
            // already kept than to the node itself. That neighbour already
            // leads towards it, so the link is better spent on another
            // direction. Pruned candidates are not added back (as in hnswlib),
            // so a node may end up with fewer than m links.
            for (uint32_t kept : ids) {
                if (distance(vector(candidate.id), vector(kept)) < candidate.distance) {
                    keep = false;
                    break;
                }
            }
        }
        if (keep) ids.push_back(candidate.id);
    }
    return ids;
}

std::vector<Neighbor> Index::search_layer(const float* query,
                                          const std::vector<uint32_t>& entry_points,
                                          size_t ef, int layer) const {
    // One flag per node, allocated per call: O(n) memory to set up, which is
    // negligible next to the distance computations at this index size. A pool
    // of reusable visited lists would remove it for much larger indexes.
    std::vector<bool> visited(size(), false);
    // candidates: nodes whose neighbours are still to be explored, closest on top.
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> candidates;
    // results: the best `ef` nodes so far, furthest on top so it is cheap to evict.
    std::priority_queue<Neighbor> results;

    for (uint32_t entry : entry_points) {
        if (visited[entry]) continue;
        visited[entry] = true;
        const Neighbor node{distance(query, vector(entry)), entry};
        candidates.push(node);
        results.push(node);
        if (results.size() > ef) results.pop();
    }

    while (!candidates.empty()) {
        const Neighbor closest = candidates.top();
        // Every remaining candidate is at least this far away, so if this one
        // cannot improve the results, none can.
        if (closest.distance > results.top().distance) break;
        candidates.pop();

        for (uint32_t neighbor : links_[closest.id][layer]) {
            if (visited[neighbor]) continue;
            visited[neighbor] = true;
            const float d = distance(query, vector(neighbor));
            if (results.size() < ef || d < results.top().distance) {
                candidates.push({d, neighbor});
                results.push({d, neighbor});
                if (results.size() > ef) results.pop();
            }
        }
    }

    // The heap yields furthest first; reverse to return closest first.
    std::vector<Neighbor> found;
    found.reserve(results.size());
    while (!results.empty()) {
        found.push_back(results.top());
        results.pop();
    }
    std::reverse(found.begin(), found.end());
    return found;
}

uint32_t Index::add(const float* vector) {
    const int level = random_level();
    const uint32_t id = store(vector, level);
    // Use the stored copy from here on: `vector` belongs to the caller.
    const float* point = this->vector(id);

    if (id == 0) {
        entry_point_ = id;
        max_level_ = level;
        return id;
    }

    // Phase 1: on the layers above the new node's level, just walk greedily
    // (ef = 1) towards it to find a good entry point for the layers below.
    std::vector<uint32_t> entry_points{entry_point_};
    for (int layer = max_level_; layer > level; --layer) {
        entry_points = {search_layer(point, entry_points, 1, layer).front().id};
    }

    // Phase 2: on each layer the node belongs to, find candidates and link.
    for (int layer = std::min(level, max_level_); layer >= 0; --layer) {
        const std::vector<Neighbor> candidates =
            search_layer(point, entry_points, ef_construction_, layer);
        const std::vector<uint32_t> chosen = select_neighbors(candidates, M_);

        links_[id][layer] = chosen;
        for (uint32_t neighbor : chosen) {
            std::vector<uint32_t>& links = links_[neighbor][layer];
            links.push_back(id);
            if (links.size() > max_degree(layer)) {
                // Too many links: keep only the closest ones, measured from
                // the neighbour. This can drop the link back to the new node,
                // so links are not always bidirectional.
                std::vector<Neighbor> ranked;
                ranked.reserve(links.size());
                for (uint32_t other : links) {
                    ranked.push_back({distance(this->vector(neighbor), this->vector(other)), other});
                }
                links = select_neighbors(std::move(ranked), max_degree(layer));
            }
        }

        // All candidates, not only the chosen ones, seed the next layer down.
        entry_points.clear();
        for (const Neighbor& candidate : candidates) entry_points.push_back(candidate.id);
    }

    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = id;
    }
    return id;
}

std::vector<Neighbor> Index::search(const float* query, size_t k, size_t ef) const {
    if (size() == 0) return {};

    std::vector<uint32_t> entry_points{entry_point_};
    for (int layer = max_level_; layer >= 1; --layer) {
        entry_points = {search_layer(query, entry_points, 1, layer).front().id};
    }

    std::vector<Neighbor> found = search_layer(query, entry_points, std::max(ef, k), 0);
    if (found.size() > k) found.resize(k);
    return found;
}

}  // namespace hnsw
