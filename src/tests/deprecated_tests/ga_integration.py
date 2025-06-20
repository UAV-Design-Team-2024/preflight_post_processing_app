import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

# [START import]
import numpy as np
# sys.path.append(r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app")
# from src.tools.point_cloud_generator import make_points, get_distance_matrix, make_final_plot, get_coord_matrix
from src.tools.field_preprocessing.field_processing import PointFactory, get_distance_matrix

import networkx as nx
import random


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

def two_opt(route, G, invalid_penalty_cost, tol):
    """
    Improves the route using 2-opt to swap segments and reduce total distance.
    """
    best_route = route
    best_distance = fitness(route, G, invalid_penalty_cost)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_distance = fitness(new_route, G, invalid_penalty_cost)

                distance_difference = new_distance - best_distance
                percent_change = distance_difference/best_distance

                if percent_change < -tol/100:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True  # Keep optimizing

    return best_route
def parallel_fitness_evaluation(population, G, invalid_penalty_cost):
    with multiprocessing.Pool(processes=5) as pool:
        fitness_values = pool.starmap(fitness, [(ind, G, invalid_penalty_cost) for ind in population])
    return fitness_values
def fitness(route, G, invalid_penalty_cost):
    total_distance = 0

    turn_penalty_cost = 1e6
  # Additional cost for sharp direction changes

    for i in range(len(route) - 1):
        if G.has_edge(route[i], route[i + 1]):
            total_distance += G[route[i]][route[i + 1]]["weight"]
        else:
            return invalid_penalty_cost
    # Penalize unnecessary turns
    for i in range(1, len(route) - 1):
        prev, curr, next_ = route[i-1], route[i], route[i+1]
        prev_coord = G.nodes[prev]["pos"]
        curr_coord = G.nodes[curr]["pos"]
        next_coord = G.nodes[next_]["pos"]

        prev_vector = (curr_coord[0] - prev_coord[0], curr_coord[1] - prev_coord[1])
        next_vector = (next_coord[0] - curr_coord[0], next_coord[1] - curr_coord[1])
        # Check if there is a sharp direction change
        if abs(prev_vector[0] - next_vector[0]) > 0.0005 and abs(prev_vector[1] - next_vector[1]) > 0.0005:
            total_distance += turn_penalty_cost  # Penalize unnecessary turns

    # Go back to starting node for each route
    # if route[0] in all_possible_paths[route[-1]]:
    #     path_segment = all_possible_paths[route[-1]][route[0]]
    #     path.extend(path_segment)
    #     total_distance += all_possible_distances[route[-1]][route[0]]
    # else:
    #     return penalty_cost

    return total_distance

def crossover(parent1, parent2, G):
    size = len(parent1)
    start, end = sorted(random.sample(range(1, size), 2))
    child = [None] * size
    # child[0] = parent1[0] # Keeps the starting node the same so we will eventually return
    child[start:end] = parent1[start:end]
    remaining = [gene for gene in parent2 if gene not in child]
    child = [remaining.pop(0) if x is None else x for x in child]
    for i in range(1, len(child) - 1):
        if not G.has_edge(child[i-1], child[i+1]):
            child[i] = random.choice(list(G.neighbors(child[i-1])))
    return child

def mutation(child, dM, G):
    dM_random = random.random()
    if dM_random < dM:
        i, j = sorted(random.sample(range(1, len(child)), 2))  # Avoid first node
        # Swap only if they are close neighbors
        if G.has_edge(child[i - 1], child[j]) and G.has_edge(child[j], child[i + 1]):
            child[i], child[j] = child[j], child[i]

    return child

def make_population(nodes ,pop_size, G, penalty_cost):
    print("Creating population for algorithm...")
    population = []
    nodes_no_start_node = [node for node in nodes if node != 0]
    while len(population) < pop_size:
        sample = [0] + sorted(nodes_no_start_node, key=lambda x: nx.shortest_path_length(G, 0, x, weight="weight"))
        if fitness(sample, G, penalty_cost) < penalty_cost:  # Ensure full connectivity
            population.append(sample)
    return population

def genetic_algorithm_base(G, initial_solution, pop_size, generations, dM, best_criteria_num, penalty_cost,
                           minimum_percent_improvement, section_number):
    best_distances_per_generation = []
    nodes, edges = list(G.nodes), list(G.edges)
    if fitness(initial_solution, G, penalty_cost) < float("inf"):
        population = [initial_solution] + [random.sample(nodes, len(nodes)) for _ in range(pop_size - 1)]
    else:
        population = [random.sample(nodes, len(nodes)) for _ in range(pop_size)]

    print(f"Created base population for section {section_number+1}")
    tik = time.perf_counter()
    for gen_num in range(generations):
        fitted_population = parallel_fitness_evaluation(population, G, penalty_cost)
        minimized_fitted_population = [x for _, x in sorted(zip(fitted_population, population))]
        best_distance = min(fitted_population)
        best_distances_per_generation.append(best_distance)

        while len(minimized_fitted_population) < pop_size: # Main GA loop here
            print("Here")
            parent1, parent2 = random.choices(fitted_population[:50], k=2)
            child = mutation(crossover(parent1, parent2, G), dM, G) # Crossover and mutation happens here
            if fitness(child, G, penalty_cost) < penalty_cost:
                minimized_fitted_population.append(child)

        population = minimized_fitted_population
    tok = time.perf_counter()
    print(f"Completed {generations} generations in {tok-tik} ")
    print(f"Beginning two-opt for section {section_number+1}...")
    final_route = two_opt(min(population, key=lambda x: fitness(x, G, penalty_cost)), G,
                          penalty_cost, minimum_percent_improvement) # Gets the absolute best route and distance
    best_dist = fitness(final_route, G, penalty_cost) #Re fit this for good measure
    # best_path = [x + 1 for x in best_path] # Put it in node terms, not in Python terms

    return best_dist, final_route, best_distances_per_generation

def run_genetic_algorithm(section_number, distance_matrix, points, boundary_polygon, length_col,
                          pop_size,
                          generations, dM,
                          best_criteria_num, penalty_cost, minimum_percent_improvement,
                          plot_initial_solutions, plot_solutions):

    print(f"Initialized process for section {section_number + 1}")
    initial_route = create_initial_route(distance_matrix, length_col)
    # if plot_initial_solutions:
    #     make_final_plot(points, boundary_polygon, initial_route)

    distance_matrix = np.asarray(distance_matrix)
    G = nx.from_numpy_array(distance_matrix)  # Make an empty graph and put nodes in it

    point_coords = []
    for point in points:
        point_coord = (point.x, point.y)
        point_coords.append(point_coord)
    for node, (x, y) in zip(G.nodes, point_coords):  # node_positions should be a dict {node: (x, y)}
        G.nodes[node]["pos"] = (x, y)
    print(f"Beginning GA solve for section {section_number + 1}")
    best_dist, final_route, best_distances_per_generation = genetic_algorithm_base(G, initial_route,
                                                                                   pop_size,
                                                                                   generations, dM,
                                                                                   best_criteria_num, penalty_cost,
                                                                                   minimum_percent_improvement, section_number)
    # print(f"Final route was: {final_route}, with a distance of: {best_dist}")
    # if plot_solutions:
    #     make_final_plot(points, boundary_polygon, final_route)
def main():

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

    # multiprocess loop for each distance matrix in distance matrices
    for i in range(total_sections):
        run_genetic_algorithm(i, distance_matrices[i], point_lists[i], boundary_polygons[i], length_cols[i], pop_size,
                          generations, dM,
                          best_criteria_num, penalty_cost, minimum_percent_improvement,
                          plot_initial_solutions, plot_solutions)

    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     sol = pool.starmap(run_genetic_algorithm, [(i, distance_matrices[i], point_lists[i], boundary_polygons[i], length_cols[i],  pop_size,
    #                       generations, dM,
    #                       best_criteria_num, penalty_cost, minimum_percent_improvement,
    #                       plot_initial_solutions, plot_solutions) for i in range(num_sections)])



if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
    # [END program]