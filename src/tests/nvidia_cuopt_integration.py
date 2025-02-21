import os
import geopandas as gp
import matplotlib.pyplot as mpl
import numpy as np
import random
import json

import shapely.plotting
from shapely.geometry import Polygon, Point
from cuopt_thin_client import CuOptServiceClient
from src.tools.point_cloud_generator import make_points, get_distance_matrix



plot = True
submit_to_cuopt = False

kml_filepath = r'C:\Users\corde\OneDrive\Documents\QGroundControl\Missions\testfield_1.kml'
height = 4.5 # meters
spacing = 10 # meters

boundary_polygon, points, altitude = make_points(kml_filepath, height, spacing)

if submit_to_cuopt:

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
        function_id='<FUNCTION_ID_OBTAINED_FROM_NGC>'
    )

if plot:
    point_cloud = shapely.plotting.plot_points(points)
    boundary_polygon_line = shapely.plotting.plot_polygon(boundary_polygon)

    mpl.show()
