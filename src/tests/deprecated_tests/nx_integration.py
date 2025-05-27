import networkx as nx
import time

# [START import]
import numpy as np

from src.tests.point_cloud_generator import make_points, is_valid_edge_for_points

# Create a 2D grid of waypoints
def main():
    kml_filepath = r'C:\Users\corde\OneDrive\Documents\QGroundControl\Missions\testfield_1.kml'
    # kml_filepath = r"C:/Users/rohan/OneDrive - University of Cincinnati/UAV Design/preflight_post_processing_app/src/tests/testfield_1.kml"
    height = 4.5  # meters
    spacing = 10  # meters
    num_processes = 4
    num_sections = 5
    use_initial_solution = False
    boundary_polygons, point_lists, altitude, length_cols = make_points(kml_filepath, height, spacing, num_sections)

    time_list = []
    for i in range(num_sections):
        tik = time.perf_counter()

        G = nx.Graph()

        points = point_lists[i]
        point_data = []
        for point in points:
            point = (float(point.xy[0][0]), float(point.xy[1][0]))
            point_data.append(point)
            G.add_node(point)

        for (x, y) in point_data:
            neighbors = [(x + spacing, y), (x - spacing, y), (x, y + spacing), (x, y - spacing)]

            for neighbor in neighbors:
                if neighbor in G.nodes and is_valid_edge_for_points((x, y), neighbor, boundary_polygons[i]):  # Check if edge is inside polygon
                    G.add_edge((x, y), neighbor, weight=np.linalg.norm(np.array(neighbor) - np.array((x, y))))

        tok = time.perf_counter()
        time_list.append(tok - tik)
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        print(f"Created graph {i + 1} in {tok - tik} s")

        start = min(point_data, key=lambda p: np.linalg.norm(np.array(p) - np.array(point_data[0])))  # Closest to (0,0)
        goal = max(point_data, key=lambda p: np.linalg.norm(np.array(p) - np.array(point_data[-1])))

        print(f"Times for pre-processing: {time_list}")

        try:
            path = nx.astar_path(G, start, goal, weight="weight",
                                 heuristic=lambda a, b: np.linalg.norm(np.array(a) - np.array(b)))
            print("Flight Path:", path)  # Outputs the optimized list of waypoints
        except:
            print("No path found")


if __name__ == "__main__":
    main()
    # [END program]