import os
import multiprocessing
import geopandas as gp
import matplotlib.pyplot as mpl
import numpy as np
import random
import json

import shapely.plotting
from shapely.ops import split
from shapely.geometry import Polygon, Point, LineString, box
import shapely.ops as s_ops
from cuopt_thin_client import CuOptServiceClient


def latlon_to_ecef(lat_deg, lon_deg, alt_m):
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

def get_coord_matrix(points, alt):
    coords = []
    for i in range(int(len(points))):
        x1, y1, z1 = latlon_to_ecef(points[i].x, points[i].y, alt)
        coords.append((x1, y1))
    return coords
# def get_distance_matrix(points, alt):
#     distance = np.zeros((len(points), len(points)))
#     for i in range(int(len(points))):
#         x1,y1,z1 = latlon_to_ecef(points[i].x, points[i].y, alt)
#         for j in range(i+1, int(len(points))):
#             x2,y2,z2 = latlon_to_ecef(points[j].x, points[j].y, alt)
#             dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
#             distance[i][j] = dist
#     distance_matrix = distance + distance.T
#     print(distance_matrix.shape)
#     distance_matrix = distance_matrix.tolist()
#     return distance_matrix

def is_valid_edge(p1, p2, boundary_edges):
    """
    Check if a path between two points crosses the boundary edges.
    """
    line = LineString([p1.coords[0], p2.coords[0]])

    # Edge is valid if it does NOT intersect any boundary edge
    for boundary_edge in boundary_edges:
        if line.intersects(boundary_edge) and not line.touches(boundary_edge):
            return False
    return True


def generate_valid_splits(boundary_polygon, num_splits, max_attempts=5, shift_step=0.1):
    """
    Generates vertical lines that only cross two boundary edges.
    Retries failed lines by shifting left or right.
    """
    minx, miny, maxx, maxy = boundary_polygon.bounds
    step_size = (maxx - minx) / (num_splits + 1)  # Ensure splits stay inside
    valid_splits = []

    # Extract boundary edges
    boundary_edges = [LineString([boundary_polygon.exterior.coords[i], boundary_polygon.exterior.coords[i + 1]])
                      for i in range(len(boundary_polygon.exterior.coords) - 1)]

    print("Attempting to split...")
    for i in range(1, num_splits + 1):
        x_coord = minx + i * step_size
        attempts = 0
        print(f"Attempt {attempts+1}")
        while attempts < max_attempts:
            vertical_line = LineString([(x_coord, miny - 1), (x_coord, maxy + 1)])  # Extend beyond bounds

            # Count how many boundary edges this line intersects
            intersections = [edge for edge in boundary_edges if vertical_line.intersects(edge)]

            if len(intersections) == 2:
                valid_splits.append(vertical_line)
                break  # Stop shifting once a valid line is found
            else:
                # Retry by shifting slightly left or right
                shift_direction = -1 if attempts % 2 == 0 else 1  # Alternate left and right
                x_coord += shift_direction * shift_step
                attempts += 1

    split_polygons = [s_ops.split(boundary_polygon, valid_split) for valid_split in valid_splits]
    print("Managed to split, continuing...")
    return split_polygons


def get_distance_row(x, y, z, row_index, points, boundary_edges):
    distances = np.array([
        np.sqrt((x[i] - x[row_index]) ** 2 + (y[i] - y[row_index]) ** 2 + (z[i] - z[row_index]) ** 2)
        if is_valid_edge(points[row_index], points[i], boundary_edges) else 1e6 for i in range(row_index + 1, len(x))
    ])

    row = np.concatenate((np.zeros(row_index + 1), distances))
    return row

def get_distance_matrix(points, alt, num_processes, boundary_polygon):
    """Generates a matrix in parallel using multiprocessing."""
    px = np.array([point.x for point in points])
    py = np.array([point.y for point in points])
    altitude = np.array([alt for _ in range(len(points))])

    x, y, z = latlon_to_ecef(px, py, altitude)
    num_rows = len(points)

    # Generate boundary edges as LineString objects
    boundary_coords = list(boundary_polygon.exterior.coords)
    boundary_edges = [LineString([boundary_coords[i], boundary_coords[i + 1]]) for i in range(len(boundary_coords) - 1)]

    # Use multiprocessing for speed
    with multiprocessing.Pool(processes=num_processes) as pool:
        rows = pool.starmap(
            get_distance_row,
            [(x, y, z, i, points, boundary_edges) for i in range(num_rows)]
        )

    # Make the matrix symmetric
    distance_matrix = np.array(rows)
    distance_matrix = (distance_matrix + distance_matrix.T).tolist()

    return distance_matrix

def get_distance_matrix_nonparallel(points, alt, num_processes, boundary_polygon):
    """Generates a matrix in parallel using multiprocessing."""
    px = np.array([point.x for point in points])
    py = np.array([point.y for point in points])
    altitude = np.array([alt for _ in range(len(points))])

    x, y, z = latlon_to_ecef(px, py, altitude)
    num_rows = len(points)

    # Generate boundary edges as LineString objects
    boundary_coords = list(boundary_polygon.exterior.coords)
    boundary_edges = [LineString([boundary_coords[i], boundary_coords[i + 1]]) for i in range(len(boundary_coords) - 1)]

    # Use multiprocessing for speed
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     rows = pool.starmap(
    rows = []
    for i in range(num_rows):
        rows.append(get_distance_row(x, y, z, i, points, boundary_edges))
        # )

    print("Generating distance matrix...")
    # Make the matrix symmetric
    distance_matrix = np.array(rows)
    distance_matrix = (distance_matrix + distance_matrix.T).tolist()

    return distance_matrix
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def divide_boundary_polygon(boundary_polygon, cols):
    minx, miny, maxx, maxy = boundary_polygon.bounds
    width = (maxx - minx) / cols
    height = (maxy - miny)

    grid_cells = [box(minx + i * width, miny + height, minx + (i + 1) * width, miny * height) for i in range(cols)]
    split_polygons = [cell.intersection(boundary_polygon) for cell in grid_cells if cell.intersects(boundary_polygon)]

    return split_polygons

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

    num_pointsx = (maxx - minx)*np.pi/180*(6378137.0 + altitude) / spacing
    num_pointsy = (maxy - miny)*np.pi/180*(6378137.0 + altitude) / spacing
    print('Number of Rows:', round(num_pointsx))
    print('Number of Columns:', round(num_pointsy))
    xrange = np.linspace(minx, maxx, num=round(num_pointsx))
    yrange = np.linspace(miny, maxy, num=round(num_pointsy))
    points = [Point(x, y) for x in xrange for y in yrange if polygon.contains(Point(x, y))]

    print(f'Number of points: {len(points)}')
    print(f'Maximum number of points: {round(num_pointsx)*round(num_pointsy)}')

    return points


def make_points(filepath, height, spacing, num_sections):
    kml_file = gp.read_file(f'{filepath}', layer='QGroundControl Plan KML')

    altitude = np.array(kml_file['geometry'][0].coords)[0][2] + height # meters
    base_polygon = kml_file["geometry"][1]
    boundary_polygons = divide_boundary_polygon(base_polygon, num_sections)
    # boundary_polygons = generate_valid_splits(base_polygon, 3)

    point_list = []
    for boundary_polygon in boundary_polygons:
        points = create_points_in_polygon(boundary_polygon, spacing, altitude)
        point_list.append(points)
        shapely.plotting.plot_polygon(boundary_polygon)
        shapely.plotting.plot_points(points)
    # print(point_list[0])
    # shapely.plotting.plot_polygon(base_polygon)
    mpl.show()

    return boundary_polygons, point_list, altitude

def make_final_plot(points=None, boundary_polygon=None, path=None):
    """
    Plots the points, boundary polygon, and path on a map.
    """
    fig, ax = mpl.subplots()
    if points:
        point_cloud = shapely.plotting.plot_points(points)
        mpl.plot(points[0].x, points[0].y, 'og', label="Starting Point")
    if boundary_polygon:
        boundary_polygon_line = shapely.plotting.plot_polygon(boundary_polygon, add_points=False)
    if path:
        path_output = LineString([points[i] for i in path])
        # path_line = shapely.plotting.plot_line(path_output)
        plot_line_with_arrows(path_output, ax)
    ax.set_title('Optimized Flight Path')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    mpl.show()


def plot_line_with_arrows(line, ax, arrow_interval=1):
    """Plots a Shapely LineString with arrows at specified intervals."""
    x, y = line.xy
    x = np.asarray(x)
    y = np.asarray(y)
    ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)