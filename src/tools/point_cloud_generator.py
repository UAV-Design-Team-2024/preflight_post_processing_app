import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import geopandas as gp
import matplotlib.pyplot as mpl
import numpy as np
import random
import json

import shapely.plotting
from shapely.ops import split
from shapely.geometry import Polygon, Point, LineString, box, MultiLineString
import shapely.ops as s_ops
from shapely.ops import unary_union


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

def is_valid_edge_for_points(p1, p2, boundary):
    return boundary.contains(Point((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))

# def generate_valid_splits(boundary_polygon, num_splits, max_attempts=5, shift_step=0.1):
#     """
#     Generates vertical lines that only cross two boundary edges.
#     Retries failed lines by shifting left or right.
#     """
#     minx, miny, maxx, maxy = boundary_polygon.bounds
#     step_size = (maxx - minx) / (num_splits + 1)  # Ensure splits stay inside
#     valid_splits = []

#     # Extract boundary edges
#     boundary_edges = [LineString([boundary_polygon.exterior.coords[i], boundary_polygon.exterior.coords[i + 1]])
#                       for i in range(len(boundary_polygon.exterior.coords) - 1)]

#     print("Attempting to split...")
#     for i in range(1, num_splits + 1):
#         x_coord = minx + i * step_size
#         attempts = 0
#         print(f"Attempt {attempts+1}")
#         while attempts < max_attempts:
#             vertical_line = LineString([(x_coord, miny - 1), (x_coord, maxy + 1)])  # Extend beyond bounds

#             # Count how many boundary edges this line intersects
#             intersections = [edge for edge in boundary_edges if vertical_line.intersects(edge)]

#             if len(intersections) == 2:
#                 valid_splits.append(vertical_line)
#                 break  # Stop shifting once a valid line is found
#             else:
#                 # Retry by shifting slightly left or right
#                 shift_direction = -1 if attempts % 2 == 0 else 1  # Alternate left and right
#                 x_coord += shift_direction * shift_step
#                 attempts += 1

#     split_polygons = [s_ops.split(boundary_polygon, valid_split) for valid_split in valid_splits]
#     print("Managed to split, continuing...")
#     return split_polygons


def get_distance_row(args):
    x, y, z, row_index, points, boundary_edges = args
    distances = []
    for i in range(row_index +1, len(x)):
        if is_valid_edge(points[row_index], points[i], boundary_edges):
            base_distance = np.sqrt((x[i] - x[row_index]) ** 2 + (y[i] - y[row_index]) ** 2 + (z[i] - z[row_index]) ** 2)
            distances.append(base_distance)
        else:
            distances.append(np.inf)
    distances = np.array(distances)
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

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        rows = executor.map(get_distance_row,
            [(x, y, z, i, points, boundary_edges) for i in range(num_rows)])

    rows = list(rows)
    # rows = [item for sublist in rows for item in sublist]
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

def flatten(xss):
    return [x for xs in xss for x in xs]
def divide_boundary_polygon(boundary_polygon, cols):
    minx, miny, maxx, maxy = boundary_polygon.bounds
    width = (maxx - minx) / cols
    height = (maxy - miny)

    grid_cells = [box(minx + i * width, miny + height, minx + (i + 1) * width, miny * height) for i in range(cols)]
    split_polygons = [cell.intersection(boundary_polygon) for cell in grid_cells if cell.intersects(boundary_polygon)]
    # print(grid_cells)
    # shapely.plotting.plot_polygon(grid_cells[0])
    return split_polygons

def perform_boundary_check(polygon, column_points, previous_inds):
    omitted_points = []
    if not previous_inds:
        ind_list = []
    else:
        ind_list = previous_inds

    boundary_check = LineString([column_points[0], column_points[-1]]) # Draws a line from the beginning of the column to the end
    intersection_pts = polygon.intersection(boundary_check) # Gets the points of the intersections

    if isinstance(intersection_pts, MultiLineString): # Check to see if we have more than two boundary intersections
        section_lines = []
        for line in intersection_pts.geoms:
            section_lines.append(line)
        num_omitted_sections = len(section_lines)
        tmp_lists = [[] for _ in range(num_omitted_sections)]
        for i in range(num_omitted_sections):
            if i == 0: # First point
                for point in column_points:
                    if point.y < section_lines[i].coords[1][1]:
                        tmp_lists[i].append(point)
                        column_points.remove(point)
            else: # Subsequent points
                for point in column_points:
                    if (point.y < section_lines[i].coords[1][1]) and (point.y > section_lines[i-1].coords[1][1]):
                        tmp_lists[i].append(point)
                        column_points.remove(point)
            # print(tmp_lists)

        len_tmp_lists = [len(tmp_lists[i]) for i in range(len(tmp_lists))]

        for length in len_tmp_lists:
            count = len_tmp_lists.count(length)
            if count > 1:
                ind = ind_list[-1]
                break
            else:
                ind = np.argmax(np.array(len_tmp_lists))

        ind_list.append(ind)
        tmp_lists.pop(ind)

        omitted_points = tmp_lists
        # print(omitted_points)


    return omitted_points, column_points, ind_list
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
    # print('Number of Rows:', round(num_pointsx))
    # print('Number of Columns:', round(num_pointsy))
    xrange = np.linspace(minx, maxx, num=round(num_pointsx))
    yrange = np.linspace(miny, maxy, num=round(num_pointsy))
    # points = [Point(x, y) for x in xrange for y in yrange if polygon.contains(Point(x, y))]
    previous_inds = None
    points = []
    column_points = []
    length_cols = []
    total_omitted_points = []


    for x in xrange:
        count = 0
        for y in yrange:
            if polygon.contains(Point(x, y)):
                # points.append(Point(x, y))
                column_points.append(Point(x, y))
                count += 1
        if count > 0:
            length_cols.append(count) #Captures the end point of the column we're currently in

        if column_points != []:
            omitted_points, modified_column_points, new_previous_inds = perform_boundary_check(polygon, column_points, previous_inds)
            total_omitted_points.append(omitted_points)
            for point in modified_column_points:
                points.append(point)
            previous_inds = new_previous_inds
        column_points = []

    # print(length_cols)
    # print(f'Number of points: {len(points)}')
    # print(f'Maximum number of points: {round(num_pointsx)*round(num_pointsy)}')

    total_omitted_points = [list for list in total_omitted_points if list]
    total_omitted_points = (flatten(total_omitted_points))
    return points, length_cols, total_omitted_points

def create_points_on_boundary(polygon, spacing, altitude):
    boundary = polygon.boundary

    # Find the rightmost x-coordinate
    max_x = max(x for x, y, z in polygon.exterior.coords)
    y_values = [y for x, y, z in polygon.exterior.coords if x == max_x]

    # Get min and max y-values
    min_y = min(y_values)
    max_y = max(y_values)

    # Extract the right boundary (edges where x == max_x)
    right_edges = []
    coords = list(polygon.exterior.coords)

    for i in range(len(coords) - 1):  # Iterate through edges
        x1, y1, z1 = coords[i]
        x2, y2, z2 = coords[i + 1]
        if x1 == max_x and x2 == max_x:
            right_edges.append(LineString([(x1, y1), (x2, y2)]))

    # Merge right boundary edges into one line
    right_boundary = unary_union(right_edges)

    # Generate points along the right boundary
    num_points = round((max_y - min_y)*np.pi/180*(6378137.0 + altitude) / spacing)  # Number of points to generate
    points = [right_boundary.interpolate(i / (num_points - 1), normalized=True) for i in range(num_points)]
    adjusted_points = [Point(p.x - 0.00000000001, p.y) for p in points]
    sorted_points = sorted(adjusted_points, key=lambda p: p.y)
    final_points = [Point(p.x, p.y) for p in sorted_points if polygon.contains(p)]
    length_col = [len(final_points)]

    return final_points, length_col

def make_points(filepath, height, spacing, num_sections, plot_sections):
    kml_file = gp.read_file(f'{filepath}', layer='QGroundControl Plan KML')

    altitude = np.array(kml_file['geometry'][0].coords)[0][2] + height # meters
    base_polygon = kml_file["geometry"][1]
    boundary_polygons = divide_boundary_polygon(base_polygon, num_sections)
    # boundary_polygons = generate_valid_splits(base_polygon, 3)

    i = 1
    point_list = []
    length_cols = []
    omitted_point_list = []
    for boundary_polygon in boundary_polygons:
    # for i in range(len(boundary_polygons)):
        print(f"Making polygon points for section {i}")
        points, len_col, omitted_points = create_points_in_polygon(boundary_polygon, spacing, altitude)
        print(f"Omitted points: {omitted_points}")
        # points_boundary, len_col_boundary = create_points_on_boundary(boundary_polygon, spacing, altitude)
        total_points = points #+ points_boundary
        total_cols = len_col #+ len_col_boundary
        point_list.append(total_points)
        # point_list.append(points)
        # point_list.append(points_boundary)
        length_cols.append(total_cols)
        omitted_point_list.append(omitted_points)
        # omitted_point_list.append([Point(-83.453, 39.157), Point(-83.453, 39.463), Point(-83.453, 39.)])
        if plot_sections:
            shapely.plotting.plot_polygon(boundary_polygon)
            # shapely.plotting.plot_points(total_points)
            shapely.plotting.plot_points(omitted_point_list)
        i +=1
        # shapely.plotting.plot_points(points_boundary)
    # print(point_list[0])
    # shapely.plotting.plot_polygon(base_polygon)



    if plot_sections:
        mpl.show()

    return boundary_polygons, point_list, altitude, length_cols

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