import os
import geopandas as gp
import matplotlib.pyplot as mpl
import numpy as np
import random
import json

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


plot = False

kml_file = gp.read_file(f'{path}/test_data.kml', layer='QGroundControl Plan KML')
height = 4.5 # meters
altitude = np.array(kml_file['geometry'][0].coords)[0][2] + height # meters
boundary_polygon = kml_file["geometry"][1]

spacing = 10 # meters
points = create_points_in_polygon(boundary_polygon, spacing, altitude)

distance_matrix = get_distance_matrix(points, altitude)

json_data = {}

cost_matrix_data = {
    "data" : {
        1:distance_matrix
    }
}

fleet_data = {
    "vehicle_locations": [[0, 0]],
    "vehicle_ids": ["Drone-A"],
    "vehicle_types": [1],
    "capacities": [[75]],
    # "vehicle_time_windows": [
    #     [0, 100]
    # ],
    # # Vehicle can take breaks in this time window as per break duration provided
    # "vehicle_break_time_windows":[
    #     [
    #         [20, 25]
    #     ]
    # ],
    # "vehicle_break_durations": [[1]]
    # Maximum cost a vehicle can incur while delivering
    # "vehicle_max_costs": [100],
    # Maximum time a vehicle can be working
    # "vehicle_max_times": [120]
}


json_data["cost_matrix_data"] = cost_matrix_data
json_data["fleet_data"] = fleet_data

sak = r"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"

cuopt_service_client = CuOptServiceClient(
    sak=sak,
    function_id=<FUNCTION_ID_OBTAINED_FROM_NGC>
)

if plot:
    point_cloud = shapely.plotting.plot_points(points)
    boundary_polygon_line = shapely.plotting.plot_polygon(boundary_polygon)

    mpl.show()
