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
from shapely.geometry import Polygon, Point, LineString, box, MultiLineString, MultiPolygon
import shapely.ops as s_ops
from shapely.ops import unary_union


def latlon_to_ecef(lat_deg, lon_deg, alt_m):
    """
    Converts latitude, longitude (in degrees), and altitude (in meters) to
    ECEF coordinates.
    """
    # WGS 84 ellipsoid parameters
    a = 6378137.0  # semi-major axis (equatorial radius in meters)
    e2 = 0.00669438002290  # eccentricity squared

    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

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

class PointFactory():
    def __init__(self, kml_filepath, spacing, height, num_sections):
        self.kml_filepath: str = kml_filepath
        self.kml_file = gp.read_file(f'{self.kml_filepath}', layer='QGroundControl Plan KML')

        self.spacing: float = spacing
        self.height: float = height
        self.altitude = np.array(self.kml_file['geometry'][0].coords)[0][2] + self.height

        self.path_direction: int = 0 # Vertical
        self.num_sections: int = num_sections
        self.previous_omitted_sections: int = 0
        self.current_omitted_sections: int = 0

        self.base_boundary_polygon: Polygon = self.kml_file["geometry"][1]

        self.boundary_polygons: list = []
        self.point_list: list = []
        self.length_cols: list = []

        self.ind: list = []
        self.column_points: list = []
        self.subsections: dict = {}
        self.omitted_points: list = []
        self.ind_list: list = []

        self.plot_sections: bool = True

        self.plotter = PlottingFactory()


    def divide_boundary_polygon(self, boundary_polygon, cols):
        minx, miny, maxx, maxy = boundary_polygon.bounds
        width = (maxx - minx) / cols
        height = (maxy - miny)

        grid_cells = [box(minx + i * width, miny + height, minx + (i + 1) * width, miny * height) for i in range(cols)]
        split_polygons = [cell.intersection(boundary_polygon) for cell in grid_cells if
                          cell.intersects(boundary_polygon)]
        # print(grid_cells)
        # shapely.plotting.plot_polygon(grid_cells[0])
        return split_polygons

    def perform_boundary_check(self, polygon, column_points):
        # print("New column")
        omitted_points = []
        ind_list = self.ind_list

        boundary_check = LineString(
            [column_points[0], column_points[-1]])  # Draws a line from the beginning of the column to the end
        intersection_pts = polygon.intersection(boundary_check)  # Gets the points of the intersections

        if isinstance(intersection_pts,
                      MultiLineString):  # Check to see if we have more than two boundary intersections
            section_lines = []
            for line in intersection_pts.geoms:
                section_lines.append(line)
            self.current_omitted_sections = len(section_lines)
            # print(num_omitted_sections)
            tmp_lists = [[] for _ in range(self.current_omitted_sections)]
            for point in column_points:
                for i in range(self.current_omitted_sections):
                    if (point.y <= section_lines[i].coords[1][1]) and (point.y > section_lines[i].coords[0][1]):
                        tmp_lists[i].append(point)

                # print(sections)
                # print(tmp_lists)

            len_tmp_lists = [len(tmp_lists[i]) for i in range(len(tmp_lists))]
            print(len_tmp_lists)
            for length in len_tmp_lists:
                count = len_tmp_lists.count(length)
                print(f"Length: {length}| Count: {count} | IndList: {ind_list}")
                if count > 1:
                    ind = ind_list[-1]
                    break
                else:
                    ind = np.argmax(np.array(len_tmp_lists))

            ind_list.append(ind)
            tmp_lists.pop(ind)

            for i in range(len(tmp_lists)):

                if not self.subsections.get(i):
                    # print("False")
                    self.subsections[i] = tmp_lists[i]
                else:
                    # print("True")
                    for val in tmp_lists[i]:
                        self.subsections[i].append(val)


            omitted_points = tmp_lists
            # print(omitted_points)

            self.ind_list = ind_list
            self.previous_omitted_sections = self.current_omitted_sections
        return omitted_points, column_points

    def create_points_in_polygon(self, polygon, spacing, altitude):

        """
        Generates random points within a given polygon.

        Args:
            polygon: A shapely Polygon object representing the boundary.
            num_points: The number of random points to generate.

        Returns:
            A list of shapely Point objects within the polygon.
        """
        minx, miny, maxx, maxy = polygon.bounds

        num_pointsx = (maxx - minx) * np.pi / 180 * (6378137.0 + altitude) / spacing
        num_pointsy = (maxy - miny) * np.pi / 180 * (6378137.0 + altitude) / spacing
        # print('Number of Rows:', round(num_pointsx))
        # print('Number of Columns:', round(num_pointsy))
        xrange = np.linspace(minx, maxx, num=round(num_pointsx))
        yrange = np.linspace(miny, maxy, num=round(num_pointsy))

        points = []
        column_points = []
        length_cols = []

        for x in xrange:
            count = 0
            for y in yrange:
                if polygon.contains(Point(x, y)):
                    # points.append(Point(x, y))
                    column_points.append(Point(x, y))
                    count += 1
            if count > 0:
                length_cols.append(count)  # Captures the end point of the column we're currently in

            if column_points != []:
                omitted_points, modified_column_points = self.perform_boundary_check(
                    polygon, column_points)
                self.omitted_points.append(omitted_points)
                for point in modified_column_points:
                    points.append(point)

            column_points = []

        # print(new_sections)
        # print(length_cols)
        # print(f'Number of points: {len(points)}')
        # print(f'Maximum number of points: {round(num_pointsx)*round(num_pointsy)}')

        # total_omitted_points = [list for list in total_omitted_points if list]
        # total_omitted_points = (flatten(total_omitted_points))
        return points, length_cols

    def create_points_on_boundary(self, polygon, spacing, altitude):
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
        num_points = round(
            (max_y - min_y) * np.pi / 180 * (6378137.0 + altitude) / spacing)  # Number of points to generate
        points = [right_boundary.interpolate(i / (num_points - 1), normalized=True) for i in range(num_points)]
        adjusted_points = [Point(p.x - 0.00000000001, p.y) for p in points]
        sorted_points = sorted(adjusted_points, key=lambda p: p.y)
        final_points = [Point(p.x, p.y) for p in sorted_points if polygon.contains(p)]
        length_col = [len(final_points)]

        return final_points, length_col

    def make_points(self):

        self.boundary_polygons = self.divide_boundary_polygon(self.base_boundary_polygon, self.num_sections)

        i = 1
        for boundary_polygon in self.boundary_polygons:
            print(f"Making polygon points for section {i}")
            points, len_col = self.create_points_in_polygon(boundary_polygon, self.spacing, self.altitude)
            # points_boundary, len_col_boundary = create_points_on_boundary(boundary_polygon, spacing, altitude)
            self.point_list.append(points) # + points_boundary
            self.length_cols.append(len_col)# + len_col_boundary
            i += 1

    def plot_points(self, show_usable=True, show_omitted=False):
        for boundary_polygon in self.boundary_polygons:
            self.plotter.add_object_to_plot(boundary_polygon)
            if show_usable:
                for val in self.point_list:
                    self.plotter.add_object_to_plot(val)
            if show_omitted:
                for val in self.subsections.values():
                    self.plotter.add_object_to_plot(val)
        self.plotter.show_plots()

class DistanceFactory():
    def __init__(self):
        self.distance_matrix: list = []
        pass

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    def is_valid_edge(self, p1, p2, boundary_edges):
        """
        Check if a path between two points crosses the boundary edges.
        """
        line = LineString([p1.coords[0], p2.coords[0]])

        # Edge is valid if it does NOT intersect any boundary edge
        for boundary_edge in boundary_edges:
            if line.intersects(boundary_edge) and not line.touches(boundary_edge):
                return False
        return True

    def get_distance_row(self, args):
        x, y, z, row_index, points, boundary_edges = args
        distances = []
        for i in range(row_index + 1, len(x)):
            if self.is_valid_edge(points[row_index], points[i], boundary_edges):
                base_distance = np.sqrt(
                    (x[i] - x[row_index]) ** 2 + (y[i] - y[row_index]) ** 2 + (z[i] - z[row_index]) ** 2)
                distances.append(base_distance)
            else:
                distances.append(np.inf)
        distances = np.array(distances)
        row = np.concatenate((np.zeros(row_index + 1), distances))
        return row

    def get_distance_matrix(self, points, alt, num_processes, boundary_polygon):
        """Generates a matrix in parallel using multiprocessing."""
        px = np.array([point.x for point in points])
        py = np.array([point.y for point in points])
        altitude = np.array([alt for _ in range(len(points))])

        x, y, z = latlon_to_ecef(px, py, altitude)
        num_rows = len(points)

        # Generate boundary edges as LineString objects
        boundary_coords = list(boundary_polygon.exterior.coords)
        boundary_edges = [LineString([boundary_coords[i], boundary_coords[i + 1]]) for i in
                          range(len(boundary_coords) - 1)]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            rows = executor.map(self.get_distance_row,
                                [(x, y, z, i, points, boundary_edges) for i in range(num_rows)])

        rows = list(rows)
        # rows = [item for sublist in rows for item in sublist]
        # Make the matrix symmetric
        distance_matrix = np.array(rows)
        distance_matrix = (distance_matrix + distance_matrix.T).tolist()
        self.distance_matrix = distance_matrix
        # return distance_matrix



class PlottingFactory():
    def __init__(self):
        pass

    def add_object_to_plot(self, object):
        if isinstance(object, Polygon) or isinstance(object, MultiPolygon):
            shapely.plotting.plot_polygon(object)

        if isinstance(object, list):
            shapely.plotting.plot_points(object)

    def make_final_plot(self, points=None, boundary_polygon=None, path=None):
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
            self.plot_line_with_arrows(path_output, ax)
        ax.set_title('Optimized Flight Path')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')

        self.show_plots()

    def plot_line_with_arrows(self, line, ax, arrow_interval=1):
        """Plots a Shapely LineString with arrows at specified intervals."""
        x, y = line.xy
        x = np.asarray(x)
        y = np.asarray(y)
        ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1)

    def show_plots(self):
        mpl.show()