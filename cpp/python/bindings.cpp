// Python bindings: recsys._hnsw.Index
#include <cstring>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "hnsw/hnsw.h"

namespace py = pybind11;

namespace {

// Accepts any array-like; converts to contiguous float32 only if needed.
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

hnsw::Metric parse_metric(const std::string& name) {
    if (name == "l2") return hnsw::Metric::L2;
    if (name == "ip") return hnsw::Metric::InnerProduct;
    throw std::invalid_argument("metric must be 'l2' or 'ip'");
}

}  // namespace

PYBIND11_MODULE(_hnsw, m) {
    py::class_<hnsw::Index>(m, "Index")
        .def(py::init([](size_t dim, const std::string& metric, size_t M,
                         size_t ef_construction, uint64_t seed, bool heuristic) {
                 return hnsw::Index(dim, parse_metric(metric), M, ef_construction, seed,
                                    heuristic);
             }),
             py::arg("dim"), py::arg("metric") = "ip", py::arg("M") = 16,
             py::arg("ef_construction") = 200, py::arg("seed") = 0,
             py::arg("heuristic") = false)
        .def("add",
             [](hnsw::Index& self, const FloatArray& vectors) {
                 if (vectors.ndim() != 2 || static_cast<size_t>(vectors.shape(1)) != self.dim()) {
                     throw std::invalid_argument("expected an array of shape (n, dim)");
                 }
                 const float* data = vectors.data();
                 const size_t n = static_cast<size_t>(vectors.shape(0));
                 // Building is pure C++; let other Python threads run meanwhile.
                 py::gil_scoped_release release;
                 for (size_t i = 0; i < n; ++i) self.add(data + i * self.dim());
             },
             py::arg("vectors"), "Insert the rows of an (n, dim) array; ids follow row order.")
        .def("search",
             [](const hnsw::Index& self, const FloatArray& query, size_t k, size_t ef) {
                 if (query.ndim() != 1 || static_cast<size_t>(query.shape(0)) != self.dim()) {
                     throw std::invalid_argument("expected a query of shape (dim,)");
                 }
                 std::vector<hnsw::Neighbor> found;
                 {
                     py::gil_scoped_release release;
                     found = self.search(query.data(), k, ef);
                 }
                 py::array_t<uint32_t> ids(found.size());
                 py::array_t<float> distances(found.size());
                 for (size_t i = 0; i < found.size(); ++i) {
                     ids.mutable_at(i) = found[i].id;
                     distances.mutable_at(i) = found[i].distance;
                 }
                 return py::make_tuple(ids, distances);
             },
             py::arg("query"), py::arg("k"), py::arg("ef"),
             "Returns (ids, distances), closest first. For 'ip' the distance is -dot.")
        .def("__len__", &hnsw::Index::size)
        .def_property_readonly("dim", &hnsw::Index::dim)
        .def_property_readonly("max_level", &hnsw::Index::max_level);
}
