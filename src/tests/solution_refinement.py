import multiprocessing
import time
import random
from concurrent.futures import ProcessPoolExecutor

# [START import]
import numpy as np
import math
import sys
# sys.path.append(r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app")
from src.tools.point_cloud_generator import make_points, get_distance_matrix, make_final_plot, get_coord_matrix

import networkx as nx
import random
import matplotlib.pyplot as mpl
def create_distance_matrices(args):
    i, points, altitude, num_processes, boundary_polygon = args
    print(f"Getting distance matrix for section {i+1}")
    distance_matrix = get_distance_matrix(points, altitude, num_processes, boundary_polygon)
    return distance_matrix

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

def route_refinement(i, distance_matrix, points, boundary_polygon, length_col, plot_initial_solutions):
    initial_route = create_initial_route(distance_matrix, length_col)
    if plot_initial_solutions:
        make_final_plot(points, boundary_polygon, initial_route)

    distance_matrix = np.asarray(distance_matrix)

def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    # [START data]

    kml_filepath = r'C:\Users\corde\OneDrive\Documents\QGroundControl\Missions\testfield_1.kml'
    # kml_filepath = r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app/src/tests/testfield_1.kml"
    height = 4.5  # meters
    spacing = 20 # meters
    num_processes = 5
    num_sections = 5

    pop_size = 100
    generations = 100
    dM = 0.1
    best_criteria_num = 10

    minimum_percent_improvement = 2 # in percent
    penalty_cost = float('inf')

    plot_sections = True
    plot_initial_solutions = False
    plot_solutions = False
    boundary_polygons, point_lists, altitude, length_cols = make_points(kml_filepath, height, spacing, num_sections, plot_sections)

    # distance_matrices = []
    #
    # with ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     tik = time.perf_counter()
    #
    #     result = list(executor.map(create_distance_matrices, [(i, point_lists[i], altitude, num_processes,
    #                                                         boundary_polygons[i]) for i in range(num_sections)]))
    #     tok = time.perf_counter()
    #     prep_time = tok - tik
    #
    #     for distance_matrix in result:
    #         distance_matrices.append(distance_matrix)
    #
    #
    #     # print(list(time_and_dist_matr))
    #
    # print(f"# of distance matrices: {len(distance_matrices)}")
    # print(f"Times for pre-processing: {prep_time}")

    # multiprocess loop for each distance matrix in distance matrices
    # for i in range(num_sections):
    #     route_refinement(i, distance_matrices[i], point_lists[i], boundary_polygons[i], length_cols[i], plot_initial_solutions)
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()