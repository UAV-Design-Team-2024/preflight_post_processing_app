import multiprocessing
import time
import random
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import networkx as nx
import matplotlib.pyplot as mpl
import pygad
import sys
sys.path.append(r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app")
from src.tools.field_processing import PointFactory, PlottingFactory, get_distance_matrix
from functools import partial


def create_distance_matrices(args):
    i, points, altitude, num_processes, boundary_polygon = args
    print(f"Getting distance matrix for section {i+1}")
    return get_distance_matrix(points, altitude, num_processes, boundary_polygon)


def create_initial_route(distance_matrix, length_cols):
    init_route = np.arange(0, len(distance_matrix), 1).tolist()
    start = 0
    init_routes = []
    for size in length_cols:
        init_routes.append(init_route[start:start+size])
        start += size
    for i in range(1, len(init_routes), 2):
        init_routes[i].reverse()
    return [point for sublist in init_routes for point in sublist]


def fitness(route, G, invalid_penalty_cost=np.inf):
    total_distance = 0
    turn_penalty_cost = 1e6

    for i in range(len(route) - 1):
        if G.has_edge(route[i], route[i + 1]):
            total_distance += G[route[i]][route[i + 1]]['weight']
        else:
            return invalid_penalty_cost

    for i in range(1, len(route) - 1):
        prev, curr, next_ = route[i-1], route[i], route[i+1]
        prev_coord, curr_coord, next_coord = G.nodes[prev]['pos'], G.nodes[curr]['pos'], G.nodes[next_]['pos']
        prev_vector = (curr_coord[0] - prev_coord[0], curr_coord[1] - prev_coord[1])
        next_vector = (next_coord[0] - curr_coord[0], next_coord[1] - curr_coord[1])
        if abs(prev_vector[0] - next_vector[0]) > 0.0005 and abs(prev_vector[1] - next_vector[1]) > 0.0005:
            total_distance += turn_penalty_cost

    return total_distance


def two_opt(route, G, invalid_penalty_cost, tol):
    best_route = route
    best_distance = fitness(route, G, invalid_penalty_cost)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_distance = fitness(new_route, G, invalid_penalty_cost)
                if (new_distance - best_distance) / best_distance < -tol/100:
                    best_route, best_distance, improved = new_route, new_distance, True

    return best_route


def run_genetic_algorithm(section_number, distance_matrix, points, boundary_polygon, length_col,
                          pop_size, generations, dM, best_criteria_num, penalty_cost, minimum_percent_improvement,
                          plot_initial_solutions, plot_solutions):

    print(f"Initialized process for section {section_number + 1}")
    initial_route = create_initial_route(distance_matrix, length_col)
    distance_matrix = np.asarray(distance_matrix)
    G = nx.from_numpy_array(distance_matrix)

    point_coords = [(point.x, point.y) for point in points]
    for node, (x, y) in zip(G.nodes, point_coords):
        G.nodes[node]['pos'] = (x, y)

    def fitness_func(ga_instance, solution, solution_idx):
        solution = list(map(int, solution))
        return -fitness(solution, G, penalty_cost)

    fitness_func_with_params = fitness_func

    gene_space = [{'low': 0, 'high': len(points) - 1} for _ in range(len(points))]

    initial_population = []
    initial_solution = list(initial_route)
    initial_population.append(initial_solution)

    while len(initial_population) < pop_size:
        candidate = list(initial_route)
        random.shuffle(candidate)
        if fitness(candidate, G, penalty_cost) < penalty_cost:
            initial_population.append(candidate)

    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=10,
        fitness_func=fitness_func_with_params,
        sol_per_pop=pop_size,
        num_genes=len(points),
        gene_space=gene_space,
        initial_population=initial_population,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=int(dM * 100),
        allow_duplicate_genes=False,
        stop_criteria=["saturate_20"]
    )

    print(f"Running PyGAD for section {section_number + 1}...")
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    final_route = list(map(int, solution))
    print(f"PyGAD raw best distance: {-solution_fitness}")

    improved_route = two_opt(final_route, G, penalty_cost, minimum_percent_improvement)
    improved_distance = fitness(improved_route, G, penalty_cost)
    print(f"Final improved distance after two-opt: {improved_distance}")

    if plot_solutions:
        plotter = PlottingFactory()
        plotter.make_final_plot(points, boundary_polygon, improved_route)

    return improved_distance, improved_route, ga_instance.best_solutions_fitness


def main():
    # kml_filepath = r'C:\Users\corde\OneDrive\Documents\QGroundControl\Missions\testfield_1.kml'
    kml_filepath = r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app/src/tests/testfield_1.kml"
    height = 4.5  # meters
    spacing = 15 # meters
    num_processes = 4
    num_sections = 5
    pop_size = 100
    generations = 500
    dM = 0.2 # mutation rate
    best_criteria_num = 10
    minimum_percent_improvement = 2
    penalty_cost = float('inf')
    plot_initial_solutions = True
    plot_solutions = True

    point_generator = PointFactory(kml_filepath=kml_filepath, spacing=spacing, height=height, num_sections=num_sections)
    point_generator.make_points()
    point_generator.plot_points(show_usable=False, show_omitted=True)

    point_lists = point_generator.all_points
    total_sections = point_generator.total_sections
    altitude = point_generator.altitude
    boundary_polygons = point_generator.boundary_polygons
    length_cols = point_generator.all_length_cols

    distance_matrices = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tik = time.perf_counter()
        result = list(executor.map(create_distance_matrices, [
            (i, point_lists[i], altitude, num_processes, boundary_polygons[i])
            for i in range(total_sections)
        ]))
        distance_matrices.extend(result)
        print(f"Pre-processing time: {time.perf_counter() - tik:.2f} seconds")

    for i in range(total_sections):
        initial_route = point_generator.create_initial_route(distance_matrices[i], length_cols[i])
        if plot_initial_solutions:
            point_generator.plotter.make_final_plot(point_lists[i], boundary_polygons[i], initial_route)

    for i in range(total_sections):
        print(len(point_lists[i]))
        run_genetic_algorithm(i, distance_matrices[i], point_lists[i], boundary_polygons[i], length_cols[i],
                              pop_size, generations, dM, best_criteria_num, penalty_cost,
                              minimum_percent_improvement, plot_initial_solutions, plot_solutions)

    # with ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     tik = time.perf_counter()
    #     result = list(executor.map(run_genetic_algorithm, [
    #         (i, distance_matrices[i], point_lists[i], boundary_polygons[i], length_cols[i],
    #          pop_size, generations, dM, best_criteria_num, penalty_cost,
    #          minimum_percent_improvement, plot_initial_solutions, plot_solutions)
    #         for i in range(total_sections)
    #     ]))
    #     print(f"PyGAD processing time: {time.perf_counter() - tik:.2f} seconds")  
    
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
