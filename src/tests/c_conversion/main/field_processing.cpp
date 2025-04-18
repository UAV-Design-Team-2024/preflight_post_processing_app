//
// Created by corde on 4/11/2025.
//
#include <omp.h>
#include <vector>
#include <cmath>
#include <future>
#include <numeric>
#include <iostream>
#include <Dense>

#include <geos_c.h>
#include <geos/geom/Point.h>
#include <geos/geom/LineString.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/Coordinate.h>
#include <geos/geom/CoordinateSequence.h>
#include <geos/geom/Geometry.h>
#include <geos/geom/Polygon.h>

#include <gdal.h>
#include <gdal_priv.h>
#include <ogrsf_frmts.h>

using namespace std;
using namespace geos::geom;



//bool check_symmetric(const Eigen::MatrixXd& mat, double rtol = 1e-5, double atol = 1e-8) {
//    return ((mat - mat.transpose()).array().abs() <= (atol + rtol * mat.array().abs().maxCoeff())).all();
//}


bool is_valid_edge(
    double x1, double y1,
    double x2, double y2,
    const std::vector<GEOSGeometry*>& boundary_edges,
    GEOSContextHandle_t context
) {
    // Create a coordinate sequence with 2 points and 2 dimensions
    GEOSCoordSequence* seq = GEOSCoordSeq_create_r(context, 2, 2);
    GEOSCoordSeq_setX_r(context, seq, 0, x1);
    GEOSCoordSeq_setY_r(context, seq, 0, y1);
    GEOSCoordSeq_setX_r(context, seq, 1, x2);
    GEOSCoordSeq_setY_r(context, seq, 1, y2);

    // Build the LineString geometry
    GEOSGeometry* line = GEOSGeom_createLineString_r(context, seq);

    for (const auto& edge : boundary_edges) {
        char intersects = GEOSIntersects_r(context, line, edge);
        char touches    = GEOSTouches_r(context, line, edge);

        if (intersects && !touches) {
            GEOSGeom_destroy_r(context, line);
            return false;
        }
    }

    GEOSGeom_destroy_r(context, line);
    return true;
}

std::vector<double> get_distance_row(
    int row_index,
    const std::vector<GEOSGeometry*>& points,  // not directly used, but kept for signature compatibility
    const std::vector<GEOSGeometry*>& boundary_edges,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& z,
    GEOSContextHandle_t context
) {
    int n = static_cast<int>(x.size());
    std::vector<double> row(n, 0.0);

    for (int i = row_index + 1; i < n; ++i) {
        if (is_valid_edge(x[i], y[i], x[row_index], y[row_index], boundary_edges, context)) {
            double dx = x[i] - x[row_index];
            double dy = y[i] - y[row_index];
            double dz = z[i] - z[row_index];
            row[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
        } else {
            row[i] = 1e6;
        }
    }

    return row;
}

void latlon_to_ecef(
    const std::vector<double>& lat_deg,
    const std::vector<double>& lon_deg,
    const std::vector<double>& alt_m,
    std::vector<double>& x,
    std::vector<double>& y,
    std::vector<double>& z
) {
    const double a = 6378137.0;            // WGS-84 semi-major axis
    const double e2 = 0.00669438002290;    // eccentricity squared

    size_t n = lat_deg.size();
    x.resize(n);
    y.resize(n);
    z.resize(n);

    for (size_t i = 0; i < n; ++i) {
        double lat_rad = lat_deg[i] * M_PI / 180.0;
        double lon_rad = lon_deg[i] * M_PI / 180.0;

        double N = a / std::sqrt(1 - e2 * std::sin(lat_rad) * std::sin(lat_rad));

        x[i] = (N + alt_m[i]) * std::cos(lat_rad) * std::cos(lon_rad);
        y[i] = (N + alt_m[i]) * std::cos(lat_rad) * std::sin(lon_rad);
        z[i] = ((1 - e2) * N + alt_m[i]) * std::sin(lat_rad);
    }
}

std::vector<std::vector<double>> get_distance_matrix(
    const std::vector<std::pair<double, double>>& latlon,
    double alt,
    const std::vector<GEOSGeometry*>& boundary_edges,
    GEOSContextHandle_t context
) {
    size_t n = latlon.size();
    std::vector<double> lat(n), lon(n), alt_vec(n, alt);

    for (size_t i = 0; i < n; ++i) {
        lat[i] = latlon[i].second;  // y = latitude
        lon[i] = latlon[i].first;   // x = longitude
    }

    std::vector<double> x, y, z;
    latlon_to_ecef(lat, lon, alt_vec, x, y, z);

    std::vector<std::vector<double>> dist_matrix(n, std::vector<double>(n, 0.0));

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); ++i) {
        std::vector<double> row = get_distance_row(i, {}, boundary_edges, x, y, z, context);
        for (int j = i + 1; j < static_cast<int>(n); ++j) {
            dist_matrix[i][j] = row[j];
            dist_matrix[j][i] = row[j];
        }
    }

    return dist_matrix;
}
