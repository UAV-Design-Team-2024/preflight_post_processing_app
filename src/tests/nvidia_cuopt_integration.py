import os
import geopandas as gp
import matplotlib.pyplot as mpl
import numpy as np
import random

import shapely.plotting
from shapely.geometry import Polygon, Point
from cuopt_thin_client import CuOptServiceClient

path = 'src/tests'

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

def get_distance_matrix(points, alt):
    distance = np.zeros((len(points), len(points)))
    for i in range(int(len(points))):
        x1,y1,z1 = latlon_to_ecef(points[i].x, points[i].y, alt)
        for j in range(i+1, int(len(points))):
            x2,y2,z2 = latlon_to_ecef(points[j].x, points[j].y, alt)
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            distance[i][j] = dist
    distance_matrix = distance + distance.T
    return distance_matrix

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
    print(num_points)

    xrange = np.linspace(minx, maxx, num=round(num_points))
    yrange = np.linspace(miny, maxy, num=round(num_points))
    points = []
    while len(points) < num_points:
        for x in xrange:
            for y in yrange:
                ordered_point = Point(x, y)
                if polygon.contains(ordered_point):
                    points.append(ordered_point)

    return points

kml_file = gp.read_file(f'{path}/test_data.kml', layer='QGroundControl Plan KML')
height = 4.5 # meters
altitude = np.array(kml_file['geometry'][0].coords)[0][2] + height # meters
# kml_file.to_file('test_data.shp')
# kml_file.to_csv('test_data.csv')
# shp_file = gp.read_file('test_data.shp')

boundary_polygon = kml_file["geometry"][1]

spacing = 10 # meters
points = create_points_in_polygon(boundary_polygon, spacing, altitude)

distmat = get_distance_matrix(points, altitude)

point_cloud = shapely.plotting.plot_points(points)
boundary_polygon_line = shapely.plotting.plot_polygon(boundary_polygon)

mpl.show()
