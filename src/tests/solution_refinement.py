import multiprocessing
import time
import random
from concurrent.futures import ProcessPoolExecutor

# [START import]
import numpy as np
import math
import sys
# sys.path.append(r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app")
# from src.tools.point_cloud_generator import make_points, get_distance_matrix, make_final_plot, get_coord_matrix
from src.tools.field_processing import PointFactory, get_distance_matrix

import networkx as nx
import random
import matplotlib.pyplot as mpl
def create_distance_matrices(args):
    i, points, altitude, num_processes, boundary_polygon = args
    print(f"Getting distance matrix for section {i+1}")
    tik = time.perf_counter()
    distance_matrix = get_distance_matrix(points, altitude, num_processes, boundary_polygon)
    tok = time.perf_counter()
    print(f"Finished section {i+1} in {tok-tik} s")
    return distance_matrix



def route_refinement(i, distance_matrix, points, boundary_polygon, length_col, plot_initial_solutions):


    distance_matrix = np.asarray(distance_matrix)

def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    # [START data]

    kml_filepath = r'C:\Users\corde\OneDrive\Documents\QGroundControl\Missions\testfield_1.kml'
    # kml_filepath = r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app/src/tests/testfield_1.kml"
    height = 4.5  # meters
    spacing = 5 # meters
    num_processes = 4
    num_sections = 5

    pop_size = 100
    generations = 100
    dM = 0.1
    best_criteria_num = 10

    minimum_percent_improvement = 2 # in percent
    penalty_cost = float('inf')

    plot_sections = True
    plot_initial_solutions = True
    plot_solutions = False

    point_generator = PointFactory(kml_filepath=kml_filepath, spacing=spacing, height=height, num_sections=num_sections)
    point_generator.make_points()
    point_generator.plot_points(show_usable=False, show_omitted=True)
    point_generator.split_omitted_clusters()
    point_generator.plot_points(show_usable=False, show_omitted=True)




    # boundary_polygons, point_lists, altitude, length_cols = make_points(kml_filepath, height, spacing, num_sections, plot_sections)

    fitted_point_lists = point_generator.point_list
    omitted_point_lists = point_generator.omitted_points

    all_points_list = fitted_point_lists + omitted_point_lists

    num_fitted_sections = point_generator.num_fitted_sections
    num_omitted_sections = point_generator.num_omitted_sections
    total_num_sections = num_omitted_sections + num_fitted_sections

    altitude = point_generator.altitude
    base_boundary_polygon = point_generator.base_boundary_polygon
    boundary_polygons = point_generator.boundary_polygons
    length_cols = point_generator.length_cols + point_generator.omitted_length_cols

    # print(point_lists)
    # print(point_generator.omitted_points)



    # prep_time = 0
    # distance_matrices = []
    #
    # with ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     tik = time.perf_counter()
    #
    #     result = list(executor.map(create_distance_matrices, [(i, fitted_point_lists[i], altitude, num_processes,
    #                                                         boundary_polygons[i]) for i in range(num_fitted_sections)]))
    #     tok = time.perf_counter()
    #     prep_time += tok - tik
    #
    #     for distance_matrix in result:
    #         distance_matrices.append(distance_matrix)
    #
    # with ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     tik = time.perf_counter()
    #
    #     result = list(executor.map(create_distance_matrices, [(i, omitted_point_lists[i], altitude, num_processes,
    #                                                            base_boundary_polygon) for i in range(num_omitted_sections)]))
    #     tok = time.perf_counter()
    #     prep_time += tok - tik
    #
    #     for distance_matrix in result:
    #         distance_matrices.append(distance_matrix)
    #
    #     # print(list(time_and_dist_matr))
    #
    # print(f"# of distance matrices: {len(distance_matrices)}")
    # print(f"Times for pre-processing: {prep_time}")
    #
    # boundary_polygons.append(base_boundary_polygon)
    # boundary_polygons.append(base_boundary_polygon)
    # boundary_polygons.append(base_boundary_polygon)
    # # multiprocess loop for each distance matrix in distance matrices
    # for i in range(total_num_sections):
    #     initial_route = point_generator.create_initial_route(distance_matrices[i], length_cols[i])
    #     if plot_initial_solutions:
    #         point_generator.plotter.make_final_plot(all_points_list[i], boundary_polygons[i], initial_route)
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()