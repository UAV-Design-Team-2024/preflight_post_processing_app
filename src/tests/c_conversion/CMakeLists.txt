cmake_minimum_required(VERSION 3.24)
project(field_converter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
        "C:/OSGeo4W/include"
        external/eigen
        main

)

link_directories(
        "C:/OSGeo4W/lib"
)

add_executable(field_processor
        main/ortools_integration.cpp
        main/field_processing.cpp
)

target_link_libraries(field_processor geos gdal_i)