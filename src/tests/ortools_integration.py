#!/usr/bin/env python3
# Copyright 2010-2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START program]
"""Simple Vehicles Routing Problem."""

# [START import]
import numpy as np
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import sys
sys.path.append(r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app")
from src.tools.point_cloud_generator import make_points, get_distance_matrix, make_final_plot, get_coord_matrix

# [END import]


# [START data_model]
def create_data_model(distance_matrix):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = distance_matrix
    data["num_vehicles"] = 1
    # [START starts_ends]
    data["depot"] = 0
    # [END starts_ends]
    return data
    # [END data_model]


# [START solution_printer]
def return_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    max_route_distance = 0
    for vehicle_id in range(data["num_vehicles"]):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            continue
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        print(f"Plan_output: {plan_output}")
        path_seq = plan_output.split('\n')
        path_seq = (path_seq[1].split(' -> '))
        path_seq = [int(x) for x in path_seq]
        max_route_distance = max(route_distance, max_route_distance)
    print(f"Maximum of the route distances: {max_route_distance}m")
    # [END solution_printer]
    return path_seq


def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    # [START data]

    # kml_filepath = r'C:\Users\corde\OneDrive\Documents\QGroundControl\Missions\testfield_1.kml'
    kml_filepath = r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app/src/tests/testfield_1.kml"
    height = 4.5  # meters
    spacing = 15  # meters
    num_processes = 4

    boundary_polygon, points, altitude = make_points(kml_filepath, height, spacing)
    # coords = get_coord_matrix(points, altitude)
    # print(coords)
    # make_point_cloud_plot(points, boundary_polygon)
    print(f"Created {len(points)} points! Beginning distance matrix creation...")
    distance_matrix = get_distance_matrix(points, altitude, num_processes)
    # print(distance_matrix)
    print(f"Created distance matrix, creating data model...")
    data = create_data_model(distance_matrix)
    # [END data]

    # Create the routing index manager.
    # [START index_manager]
    print("Creating a manager...")
    manager = pywrapcp.RoutingIndexManager( 
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )
    # [END index_manager]

    # Create Routing Model.
    # [START routing_model]
    print("Starting routing model...")
    routing = pywrapcp.RoutingModel(manager)
    # [END routing_model]

    # Create and register a transit callback.
    # [START transit_callback]
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        callback = data["distance_matrix"][from_node][to_node]
        return int(callback)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # [END transit_callback]

    # Define cost of each arc.
    # [START arc_cost]
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # [END arc_cost]

    # Add Distance constraint.
    # [START distance_constraint]
    # dimension_name = "Distance"
    # routing.AddDimension(
    #     transit_callback_index,
    #     0,  # no slack
    #     2000,  # vehicle maximum travel distance
    #     True,  # start cumul to zero
    #     dimension_name,
    # )
    # distance_dimension = routing.GetDimensionOrDie(dimension_name)
    # distance_dimension.SetGlobalSpanCostCoefficient(100)
    # [END distance_constraint]

    # Setting first solution heuristic.
    # [START parameters]
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
    # search_parameters.time_limit.seconds = 60 * 3


    search_parameters.log_search = True
    # [END parameters]

    # Solve the problem.
    # [START solve]
    print("Beginning solve...")
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        print(f"Solution found! Printing...")
        path_seq = return_solution(data, manager, routing, solution)
        make_final_plot(points, boundary_polygon, path_seq)
    # [END return_solution]


if __name__ == "__main__":
    main()
    # [END program]