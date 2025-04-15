//
// Created by corde on 4/11/2025.
//
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

using namespace std;
using namespace geos::geom;

GeometryFactory::Ptr factory = GeometryFactory::create();

bool check_symmetric(const Eigen::MatrixXd& mat, double rtol = 1e-5, double atol = 1e-8) {
    return ((mat - mat.transpose()).array().abs() <= (atol + rtol * mat.array().abs().maxCoeff())).all();
}

bool is_valid_edge(
    double x1, double y1,
    double x2, double y2,
    const std::vector<GEOSGeometry*>& boundary_edges
) {
    // Create coordinate sequence
    GEOSCoordSequence* seq = GEOSCoordSeq_create(2, 2);
    GEOSCoordSeq_setX(seq, 0, x1);
    GEOSCoordSeq_setY(seq, 0, y1);
    GEOSCoordSeq_setX(seq, 1, x2);
    GEOSCoordSeq_setY(seq, 1, y2);

    // Build the line geometry
    GEOSGeometry* line = GEOSGeom_createLineString(seq);

    for (const auto& edge : boundary_edges) {
        char intersects = GEOSIntersects(line, edge);
        char touches = GEOSTouches(line, edge);

        if (intersects && !touches) {
            GEOSGeom_destroy(line);
            return false;
        }
    }

    GEOSGeom_destroy(line);
    return true;
}

tuple<double, double, double> latlon_to_ecef(double lat_deg, double lon_deg, double alt_m) {
    const double a = 6378137.0; // WGS-84 semi-major axis
    const double e2 = 0.00669438002290; // eccentricity squared

    double lat_rad = lat_deg * M_PI / 180.0;
    double lon_rad = lon_deg * M_PI / 180.0;

    double N = a / sqrt(1 - e2 * sin(lat_rad) * sin(lat_rad));
    double x = (N + alt_m) * cos(lat_rad) * cos(lon_rad);
    double y = (N + alt_m) * cos(lat_rad) * sin(lon_rad);
    double z = ((1 - e2) * N + alt_m) * sin(lat_rad);

    return {x, y, z};
}

vector<double> get_distance_row(
    int row_index,
    const vector<Point*>& points,
    const vector<const Geometry*>& boundary_edges,
    const vector<double>& x,
    const vector<double>& y,
    const vector<double>& z
) {
    int n = points.size();
    vector<double> row(n, 0.0);
    for (int i = row_index + 1; i < n; ++i) {
        if (is_valid_edge(points[row_index], points[i], boundary_edges)) {
            double dx = x[i] - x[row_index];
            double dy = y[i] - y[row_index];
            double dz = z[i] - z[row_index];
            row[i] = sqrt(dx * dx + dy * dy + dz * dz);
        } else {
            row[i] = 1e6;
        }
    }
    return row;
}

vector<vector<double>> get_distance_matrix(
    const vector<Point*>& points,
    double altitude,
    const Polygon* boundary_polygon,
    int num_threads
) {
    int n = points.size();
    vector<double> lat(n), lon(n), alt(n);
    for (int i = 0; i < n; ++i) {
        lat[i] = points[i]->getX();
        lon[i] = points[i]->getY();
        alt[i] = altitude;
    }

    vector<double> x(n), y(n), z(n);
    for (int i = 0; i < n; ++i) {
        tie(x[i], y[i], z[i]) = latlon_to_ecef(lat[i], lon[i], alt[i]);
    }

    // Build boundary edges
    const CoordinateSequence* coords = boundary_polygon->getExteriorRing()->getCoordinatesRO();
    vector<const Geometry*> boundary_edges;
    for (size_t i = 0; i < coords->size() - 1; ++i) {
        vector<Coordinate> edge_coords = {
            coords->getAt(i),
            coords->getAt(i + 1)
        };
        boundary_edges.push_back(factory->createLineString(edge_coords));
    }
    // Multithreading setup
    vector<future<vector<double>>> futures;
    for (int i = 0; i < n; ++i) {
        futures.push_back(async(launch::async, get_distance_row, i, cref(points), cref(boundary_edges), cref(x), cref(y), cref(z)));
    }

    // Collect rows
    vector<vector<double>> matrix(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        vector<double> row = futures[i].get();
        matrix[i] = row;
    }

    // Make symmetric
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            matrix[i][j] = matrix[j][i];
        }
    }

    return matrix;
}
