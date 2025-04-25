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
import sys
sys.path.append(r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app")
sys.path.append(r'C:\Users\corde\Documents\Projects\preflight_post_processing_app\src\tools\field_preprocessing\c_functions\cmake-build-release')
# from src.tools.point_cloud_generator import make_points, get_distance_matrix, make_final_plot, get_coord_matrix
from src.tools.field_preprocessing.field_processing import PointFactory, get_distance_matrix_c
from src.tests.output import output_pts_to_file
import field_processing_c_module
from field_processing_c_module import run_solver

def create_distance_matrices(args):
    i, points, altitude, num_processes, boundary_polygon = args
    print(f"Getting distance matrix for section {i+1}")
    tik = time.perf_counter()
    distance_matrix = get_distance_matrix_c(points, altitude, num_processes, boundary_polygon)
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


def run_ortools(distance_matrix ,use_initial_solution, init_route = None):
    if use_initial_solution:
        initial_route = init_route
        data = create_data_model(distance_matrix, initial_route)
    else:
        data = create_data_model(distance_matrix)

    path_seq = run_solver(data['distance_matrix'], data['num_vehicles'],
                          data['starts'], data['ends'])

    if path_seq:
        return path_seq
    else:
        print("No solution found OR solution is broken")

def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    # [START data]

    kml_filepath = r'C:\Users\corde\OneDrive\Documents\QGroundControl\Missions\testfield_1.kml'
    # kml_filepath = r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app/src/tests/testfield_1.kml"
    height = 4.5  # meters
    spacing = 10 # meters
    num_processes = 4
    num_sections = 5

    reinitialize = False
    use_initial_solution = False
    plot_sections = True
    plot_initial_solutions = False
    plot_solutions = False

    point_generator = PointFactory(kml_filepath=kml_filepath, spacing=spacing, height=height, num_sections=num_sections)
    point_generator.make_points()
    # point_generator.plot_points(show_usable=False, show_omitted=True)

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

    init_solutions = []
    init_path_processing_times = []

    # First Pass
    with multiprocessing.Pool(processes=num_processes) as pool:
        tik = time.perf_counter()
        sol = pool.starmap(run_ortools, [
            (distance_matrices[i], use_initial_solution) for i in
            range(total_sections)])
        tok = time.perf_counter()
        init_solutions.append(sol)
        init_path_processing_times.append(tok-tik)

    final_solutions = []
    final_path_processing_times = []

    if reinitialize:
        with multiprocessing.Pool(processes=num_processes) as pool:
            tik = time.perf_counter()
            sol = pool.starmap(run_solver, [
                (distance_matrices[i], True, init_solutions[0][i])
                for i in
                range(total_sections)])
            tok = time.perf_counter()
            final_solutions.append(sol)
            final_path_processing_times.append(tok - tik)
    else:
        final_solutions = init_solutions
        final_path_processing_times = init_path_processing_times


    # Final Pass

    print(f"Total time for field processing: {sum(final_path_processing_times)}")
    for i in range(total_sections):
        point_generator.plotter.make_final_plot(point_lists[i], boundary_polygons[i], final_solutions[0][i])

    output_pts_to_file(final_solutions, point_lists, total_sections, altitude)
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()