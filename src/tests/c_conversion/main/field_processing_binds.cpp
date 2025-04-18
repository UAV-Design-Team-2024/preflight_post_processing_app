//
// Created by corde on 4/17/2025.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <geos_c.h>
#include <field_processing.h>
#include <geos/geom/GeometryFactory.h>

namespace py = pybind11;

using namespace geos::geom;
using namespace geos::io;

extern GEOSContextHandle_t global_context = GEOS_init_r();

PYBIND11_MODULE(field_processing_c_module, m) {
    m.doc() = "High-performance field processing module using GEOS & GDAL";

//     m.def("get_distance_row", [](int row_index,
//                                 const std::vector<std::pair<double, double>>& py_points,
//                                 const std::vector<std::string>& wkt_edges,
//                                 const std::vector<double>& x,
//                                 const std::vector<double>& y,
//                                 const std::vector<double>& z) {
//
//       GEOSContextHandle_t context = global_context;
//
//       std::vector<GEOSGeometry*> cpp_points;
//       for (const auto& p : py_points) {
//           GEOSCoordSequence* cs = GEOSCoordSeq_create_r(context, 1, 2);
//           GEOSCoordSeq_setX_r(context, cs, 0, p.first);
//           GEOSCoordSeq_setY_r(context, cs, 0, p.second);
//           GEOSGeometry* pt = GEOSGeom_createPoint_r(context, cs);
//           cpp_points.push_back(pt);
//       }
//
//       GEOSWKTReader* reader = GEOSWKTReader_create_r(context);
//       std::vector<GEOSGeometry*> boundary_edges;
//
//       for (const auto& wkt : wkt_edges) {
//           try {
//               const GEOSGeometry* g = GEOSWKTReader_read_r(context, reader, wkt.c_str());
//               if (g) {
//                   boundary_edges.push_back((GEOSGeometry*)g);
//               } else {
//                   std::cerr << "[GEOS Parse Error] Failed to parse: " << wkt << "\n";
//               }
//           } catch (...) {
//               std::cerr << "[GEOS Exception] Something went wrong parsing: " << wkt << "\n";
//           }
//       }
//
//       std::vector<double> result = get_distance_row(
//           row_index, cpp_points, boundary_edges, x, y, z, context
//       );
//
//       // Cleanup
//       for (auto* g : cpp_points) GEOSGeom_destroy_r(context, g);
//       for (auto* g : boundary_edges) GEOSGeom_destroy_r(context, g);
//       GEOSWKTReader_destroy_r(context, reader);
//       GEOS_finish_r(global_context);
//       return result;
// });

    m.def("get_distance_matrix", [](
    const std::vector<std::pair<double, double>>& latlon,
    double altitude,
    const std::vector<std::string>& wkt_edges
) {
    GEOSContextHandle_t context = global_context;

    // Parse WKT edges
    GEOSWKTReader* reader = GEOSWKTReader_create_r(context);
    std::vector<GEOSGeometry*> boundary_edges;

    for (const auto& wkt : wkt_edges) {
        GEOSGeometry* g = GEOSWKTReader_read_r(context, reader, wkt.c_str());
        if (g) {
            boundary_edges.push_back(g);
        } else {
            std::cerr << "[GEOS C API] WKT parse failed: " << wkt << "\n";
        }
    }

    auto matrix = get_distance_matrix(latlon, altitude, boundary_edges, context);

    // Cleanup
    for (auto* g : boundary_edges) GEOSGeom_destroy_r(context, g);
    GEOSWKTReader_destroy_r(context, reader);

    return matrix;
});
}
