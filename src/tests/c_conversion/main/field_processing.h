//
// Created by corde on 4/11/2025.
//

#ifndef FIELD_PROCESSING_H
#define FIELD_PROCESSING_H

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

//bool check_symmetric(const Eigen::MatrixXd& mat, double rtol = 1e-5, double atol = 1e-8);

//tuple<double, double, double> latlon_to_ecef(double lat_deg, double lon_deg, double alt_m);

bool is_valid_edge(const Point* p1, const Point* p2, const vector<const Geometry*>& boundary_edges);

vector<double> get_distance_row(int row_index, const vector<Point*>& points,
                                const vector<const Geometry*>& boundary_edges,
                                const vector<double>& x, const vector<double>& y, const vector<double>& z);

//vector<vector<double>> get_distance_matrix(const vector<Point*>& points, double altitude, const Polygon* boundary_polygon, int num_threads);
