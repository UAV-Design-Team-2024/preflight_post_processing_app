//
// Created by corde on 4/17/2025.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <geos_c.h>
#include <field_processing.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/io/WKTReader.h>

namespace py = pybind11;

using namespace geos::geom;
using namespace geos::io;
PYBIND11_MODULE(field_processing, m) {
    m.doc() = "High-performance field processing module using GEOS & GDAL";

    m.def("get_distance_row", [](int row_index,
                              const std::vector<std::pair<double, double>>& py_points,
                              const std::vector<std::string>& wkt_edges,
                              const std::vector<double>& x,
                              const std::vector<double>& y,
                              const std::vector<double>& z) {
    auto factory = GeometryFactory::create();

    std::vector<Point*> cpp_points;
    for (const auto& p : py_points) {
        Coordinate coord(p.first, p.second);
        cpp_points.push_back(factory->createPoint(coord).release());
    }

    WKTReader reader(factory.get());
    std::vector<const Geometry*> boundary_edges;

    // std::string test = "A";
    //
    // try
    // {
    //     auto g = reader.read(test);
    // }
    // catch(const std::exception& ex)
    // {
    //     std::cerr << "Failed to parse: \"" << test << "\"\nReason: " << ex.what() << std::endl;
    // }
    //
    //
    // for (const auto& wkt : wkt_edges) {
    //     std::cerr << "[WKT Input] -> \"" << wkt << "\"" << std::endl;
    //     // parse here
    // }
    //
    // for (const auto& wkt : wkt_edges) {
    //     std::cout << "[DEBUG] Length: " << wkt.size() << ", Last char: " << (int)wkt.back() << "\n";
    // }

    for (const auto& wkt : wkt_edges) {
        std::string safe_wkt = std::string(wkt.c_str());
        std::cout << "[WKT Passed] " << safe_wkt << std::endl;
        std::cout << "Length: " << safe_wkt.length() << std::endl;

        try {
            std::unique_ptr<Geometry> g(reader.read(safe_wkt));
            boundary_edges.push_back(g.release());
        } catch (const geos::io::ParseException& e) {
            std::cerr << "GEOS Parse Error: " << e.what() << std::endl;
        }
    }
    std::vector<double> result = get_distance_row(row_index, cpp_points, boundary_edges, x, y, z);

    // cleanup
    for (auto* p : cpp_points) delete p;
    for (auto* g : boundary_edges) delete g;

    return result;
});

}
