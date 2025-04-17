//
// Created by corde on 4/17/2025.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <geos_c.h>
#include <field_processing.h>

namespace py = pybind11;

PYBIND11_MODULE(field_processing, m) {
    m.doc() = "High-performance field processing module using GEOS & GDAL";

    m.def("get_distance_row", &get_distance_row,
          py::arg("row_index"), py::arg("points"), py::arg("boundary_edges"),
          py::arg("x"), py::arg("y"), py::arg("z"),
          "Compute a row of the distance matrix using C++.");
}
