//
// Created by corde on 4/17/2025.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <geos_c.h>
#include <field_processing.h>
#include <ortools_integration.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <vector>

#include <ortools/constraint_solver/constraint_solver.h>
#include <ortools/constraint_solver/routing.h>
#include <ortools/constraint_solver/routing_index_manager.h>

namespace py = pybind11;




PYBIND11_MODULE(field_processing_c_module, m)
{
    m.doc() = "High-performance field processing module using GEOS & GDAL";

    GEOSContextHandle_t global_context = GEOS_init_r();

    m.def("get_distance_matrix", [global_context](
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


    m.def("run_solver", [](
    const std::vector<std::vector<double>>& distance_matrix,
                              int num_vehicles,
                              const std::vector<int>& starts,
                              const std::vector<int>& ends)
        {
            auto path_seq = SolveRouting(distance_matrix, num_vehicles, starts, ends);
            return path_seq;

        }
    );
}