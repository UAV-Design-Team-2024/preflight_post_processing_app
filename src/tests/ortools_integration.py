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
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

# [START import]
import numpy as np
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import sys
sys.path.append(r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app")
# from src.tools.point_cloud_generator import make_points, get_distance_matrix, make_final_plot, get_coord_matrix
from src.tools.field_processing import PointFactory, PlottingFactory, get_distance_matrix

def create_distance_matrices(args):
    i, points, altitude, num_processes, boundary_polygon = args
    print(f"Getting distance matrix for section {i+1}")
    tik = time.perf_counter()
    distance_matrix = get_distance_matrix(points, altitude, num_processes, boundary_polygon)
    tok = time.perf_counter()
    print(f"Finished section {i+1} in {tok-tik} s")
    return distance_matrix

def create_data_model(distance_matrix, init_route=None):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = distance_matrix
    data["num_vehicles"] = 1
    # [START starts_ends]
    # [END starts_ends]
    data["starts"] = [0]
    data["ends"] = [len(distance_matrix)-1]
    if init_route is not None:
        data["initial_routes"] = [init_route]
    #https://developers.google.com/optimization/routing/routing_tasks#:~:text=SolutionLimit(100);-,Setting%20initial%20routes%20for%20a%20search,solution%20using%20the%20method%20ReadAssignmentFromRoutes%20.
    return data
    # [END data_model]
def create_initial_route(distance_matrix, length_cols):
    init_route = np.arange(0, len(distance_matrix), 1).tolist()
    start = 0
    init_routes = []
    for size in length_cols:
        init_routes.append(init_route[start:start+size])
        start += size
    for i in range(1,len(init_routes),2):
        init_routes[i].reverse()
    finalized_init_route = [point for sublist in init_routes for point in sublist]
    return finalized_init_route


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

def run_solver(distance_matrix, points, boundary_polygon, length_col, use_initial_solution):
    if use_initial_solution:
        initial_route = create_initial_route(distance_matrix, length_col)
        data = create_data_model(distance_matrix, initial_route)
    else:
        data = create_data_model(distance_matrix)

    # [START index_manager]
    print("Creating a manager...")
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["starts"], data["ends"]
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
        distance_matrix = data["distance_matrix"]
        cost = distance_matrix[from_node][to_node]
        return int(cost)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
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
    #
    # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.first_solution_strategy = (
    #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    # )  # Prefer shortest paths
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # )  # Optimize paths iteratively
    # search_parameters.time_limit.seconds = 30
    # search_parameters.time_limit.seconds = 60 * 3
    search_parameters.log_search = True
    # [END parameters]
    routing.CloseModelWithParameters(search_parameters)
    if use_initial_solution:
        initial_solution = routing.ReadAssignmentFromRoutes(data["initial_routes"], True)
        solution = routing.SolveFromAssignmentWithParameters(initial_solution, search_parameters)
    else:
        solution = routing.SolveWithParameters(search_parameters)
    # Solve the problem.
    # [START solve]
    print("Beginning solve...")
    if solution:
        print(f"Solution found! Printing...")
        path_seq = return_solution(data, manager, routing, solution)
        plotFact = PlottingFactory()
        plotFact.make_final_plot(points, boundary_polygon, path_seq)
    else:
        print("No solution found...")
    # [END return_solution]

def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    # [START data]

    kml_filepath = r'C:\Users\corde\OneDrive\Documents\QGroundControl\Missions\testfield_1.kml'
    # kml_filepath = r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app/src/tests/testfield_1.kml"
    height = 4.5  # meters
    spacing = 15 # meters
    num_processes = 4
    num_sections = 5

    pop_size = 100
    generations = 100
    dM = 0.1
    best_criteria_num = 10

    minimum_percent_improvement = 2 # in percent
    penalty_cost = float('inf')

    use_initial_solution = False
    plot_sections = True
    plot_initial_solutions = True
    plot_solutions = False

    point_generator = PointFactory(kml_filepath=kml_filepath, spacing=spacing, height=height, num_sections=num_sections)
    point_generator.make_points()
    point_generator.plot_points(show_usable=False, show_omitted=True)


    # boundary_polygons, point_lists, altitude, length_cols = make_points(kml_filepath, height, spacing, num_sections, plot_sections)

    point_lists = point_generator.all_points


    total_sections = point_generator.total_sections

    altitude = point_generator.altitude
    base_boundary_polygon = point_generator.base_boundary_polygon
    boundary_polygons = point_generator.boundary_polygons
    length_cols = point_generator.all_length_cols


    prep_time = 0
    distance_matrices = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tik = time.perf_counter()

        result = list(executor.map(create_distance_matrices, [(i, point_lists[i], altitude, num_processes,
                                                            boundary_polygons[i]) for i in range(total_sections)]))
        tok = time.perf_counter()
        prep_time += tok - tik

        for distance_matrix in result:
            distance_matrices.append(distance_matrix)

    print(f"# of distance matrices: {len(distance_matrices)}")
    print(f"Times for pre-processing: {prep_time}")

    # multiprocess loop for each distance matrix in distance matrices
    for i in range(total_sections):
        initial_route = point_generator.create_initial_route(distance_matrices[i], length_cols[i])
        if plot_initial_solutions:
            point_generator.plotter.make_final_plot(point_lists[i], boundary_polygons[i], initial_route)

    with multiprocessing.Pool(processes=num_processes) as pool:
        sol = pool.starmap(run_solver, [
            (distance_matrices[i], point_lists[i], boundary_polygons[i], length_cols[i], use_initial_solution) for i in
            range(total_sections)])
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()