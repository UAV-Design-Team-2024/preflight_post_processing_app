//
// Created by corde on 4/11/2025.
//

#ifndef FIELD_PROCESSING_H
#define FIELD_PROCESSING_H
#include <geos_c.h>

#endif //FIELD_PROCESSING_H

#pragma once
#include <vector>
#include <tuple>
#include <Dense>
#include <geos/geom/Point.h>
#include <geos/geom/Geometry.h>
#include <geos/geom/Polygon.h>

using namespace std;
using namespace geos::geom;


tuple<double, double, double> latlon_to_ecef(const std::vector<double>& lat_deg,
    const std::vector<double>& lon_deg,
    const std::vector<double>& alt_m,
    std::vector<double>& x,
    std::vector<double>& y,
    std::vector<double>& z);

bool is_valid_edge(double x1, double y1,
    double x2, double y2,
    const std::vector<GEOSGeometry*>& boundary_edges,
    GEOSContextHandle_t context);

vector<double> get_distance_row(    int row_index,
    const std::vector<GEOSGeometry*>& points,  // not directly used, but kept for signature compatibility
    const std::vector<GEOSGeometry*>& boundary_edges,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& z,
    GEOSContextHandle_t context);

vector<vector<double>> get_distance_matrix( const std::vector<std::pair<double, double>>& latlon,
    double alt,
    const std::vector<GEOSGeometry*>& boundary_edges,
    GEOSContextHandle_t context);
