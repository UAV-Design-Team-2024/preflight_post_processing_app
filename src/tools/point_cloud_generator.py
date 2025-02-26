import os
import multiprocessing
import geopandas as gp
import matplotlib.pyplot as mpl
import numpy as np
import random
import json

import shapely.plotting
from shapely.geometry import Polygon, Point, LineString
from cuopt_thin_client import CuOptServiceClient


def latlon_to_ecef(lat_deg, lon_deg, alt_m=0):
    """
    Converts latitude, longitude (in degrees), and altitude (in meters) to
    ECEF coordinates.
    """
    # WGS 84 ellipsoid parameters
    a = 6378137.0          # semi-major axis (equatorial radius in meters)
    e2 = 0.00669438002290  # eccentricity squared

    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    x = (N + alt_m) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt_m) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((1 - e2) * N + alt_m) * np.sin(lat_rad)

    return x, y, z

def get_coord_matrix(points, alt):
    coords = []
    for i in range(int(len(points))):
        x1, y1, z1 = latlon_to_ecef(points[i].x, points[i].y, alt)
        coords.append((x1, y1))
    return coords
# def get_distance_matrix(points, alt):
#     distance = np.zeros((len(points), len(points)))
#     for i in range(int(len(points))):
#         x1,y1,z1 = latlon_to_ecef(points[i].x, points[i].y, alt)
#         for j in range(i+1, int(len(points))):
#             x2,y2,z2 = latlon_to_ecef(points[j].x, points[j].y, alt)
#             dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
#             distance[i][j] = dist
#     distance_matrix = distance + distance.T
#     print(distance_matrix.shape)
#     distance_matrix = distance_matrix.tolist()
#     return distance_matrix

def get_distance_row( x, y, z, row_index):
    distances = np.array([np.sqrt((x[i]-x[row_index])**2 + (y[i]-y[row_index])**2 + (z[i]-z[row_index])**2) for i in range(row_index+1, len(x))])
    row = np.concatenate((np.zeros(row_index+1), distances))
    # row = np.array([np.sqrt((x[i]-x[row_index])**2 + (y[i]-y[row_index])**2 + (z[i]-z[row_index])**2) for i in range(len(x))])

    return row

def get_distance_matrix(points, alt, num_processes):
    """Generates a matrix in parallel using multiprocessing."""
    px = np.array([point.x for point in points])
    py = np.array([point.y for point in points])
    altitude = np.array([alt for i in range(len(points))])
    x, y, z = latlon_to_ecef(px, py, altitude)
    num_rows = len(points)
    with multiprocessing.Pool(processes=num_processes) as pool:
        rows = pool.starmap(get_distance_row, [(x, y, z, i) for i in range(num_rows)])
    mat = np.array(rows)
    dist_mat = (mat + mat.T).tolist()
    return dist_mat

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def create_points_in_polygon(polygon, spacing, altitude):


    """
    Generates random points within a given polygon.

    Args:
        polygon: A shapely Polygon object representing the boundary.
        num_points: The number of random points to generate.

    Returns:
        A list of shapely Point objects within the polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds

    x1,y1,z1 = latlon_to_ecef(minx, miny, altitude)
    x2,y2,z2 = latlon_to_ecef(maxx, maxy, altitude)

    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    num_points = dist / spacing

    xrange = np.linspace(minx, maxx, num=round(num_points))
    yrange = np.linspace(miny, maxy, num=round(num_points))
    points = []
    for x in xrange:
        for y in yrange:
            ordered_point = Point(x, y)
            if polygon.contains(ordered_point):
                points.append(ordered_point)
    print(f'Number of points: {len(points)}')
    print(f'Maximum number of points: {round(num_points)**2}')

    return points


def make_points(filepath, height, spacing):
    kml_file = gp.read_file(f'{filepath}', layer='QGroundControl Plan KML')

    altitude = np.array(kml_file['geometry'][0].coords)[0][2] + height # meters
    boundary_polygon = kml_file["geometry"][1]


    points = create_points_in_polygon(boundary_polygon, spacing, altitude)
    return boundary_polygon, points, altitude

def make_final_plot(points=None, boundary_polygon=None, path=None):
    """
    Plots the points, boundary polygon, and path on a map.
    """
    fig, ax = mpl.subplots()
    if points:
        point_cloud = shapely.plotting.plot_points(points)
        mpl.plot(points[0].x, points[0].y, 'og', label="Starting Point")
    if boundary_polygon:
        boundary_polygon_line = shapely.plotting.plot_polygon(boundary_polygon, add_points=False)
    if path:
        path_output = LineString([points[i] for i in path])
        # path_line = shapely.plotting.plot_line(path_output)
        plot_line_with_arrows(path_output, ax)
    ax.set_title('Optimized Flight Path')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    mpl.show()


def plot_line_with_arrows(line, ax, arrow_interval=1):
    """Plots a Shapely LineString with arrows at specified intervals."""
    x, y = line.xy
    x = np.asarray(x)
    y = np.asarray(y)
    ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)