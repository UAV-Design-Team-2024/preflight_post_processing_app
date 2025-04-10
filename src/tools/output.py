import numpy as np
from shapely.geometry import Point, Polygon



def output_file(path, point_list, num_sections, relative_altitude):
    file_name = 'UAV_Mission_Plan.txt'
    with open(file_name, "w") as file:
        for i in range(num_sections):
            section_points = point_list[i]
            for j in path[0][i]:
                file.write(f'{section_points[j].x} {section_points[j].y} {relative_altitude} 0.5 0\n')

# path = [[0, 1, 2, 3, 4]]
# num_section = 1
# point_list = [[Point(1, 2), Point(3, 4), Point(5, 6), Point(7, 8), Point(9, 10)]]
# relative_altitude = 10
#
# output_file(path, point_list, num_section, relative_altitude)
