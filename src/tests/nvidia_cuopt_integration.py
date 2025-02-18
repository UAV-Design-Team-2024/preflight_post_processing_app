import os
import geopandas as gp
import matplotlib.pyplot as mpl
import numpy as np
import random
import json


import shapely.plotting
from shapely.geometry import Polygon, Point

from cuopt_thin_client import CuOptServiceClient
def create_points_in_polygon(polygon, num_points):
    """
    Generates random points within a given polygon.

    Args:
        polygon: A shapely Polygon object representing the boundary.
        num_points: The number of random points to generate.

    Returns:
        A list of shapely Point objects within the polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds

    xrange = np.linspace(minx, maxx, num=num_points)
    yrange = np.linspace(miny, maxy, num=num_points)
    points = []
    while len(points) < num_points:
        for x in xrange:
            for y in yrange:
                ordered_point = Point(x, y)
                if polygon.contains(ordered_point):
                    points.append(ordered_point)

    return points

kml_file = gp.read_file('test_data.kml', layer='QGroundControl Plan KML')
# kml_file.to_file('test_data.shp')
# kml_file.to_csv('test_data.csv')
shp_file = gp.read_file('test_data.shp')

boundary_polygon = kml_file["geometry"][1]

num_points = 75
points = create_points_in_polygon(boundary_polygon, num_points)
point_cloud = shapely.plotting.plot_points(points)
boundary_polygon_line = shapely.plotting.plot_polygon(boundary_polygon)

mpl.show()
