cmake_minimum_required(VERSION 3.24)
project(field_converter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
        "C:/OSGeo4W/include"
        "C:/Users/corde/anaconda3/envs/uc-agri-drone-env/include"
        external/eigen
        external/pybind11/include/
        main

)

link_directories(
        "C:/OSGeo4W/lib"
        "C:/Users/corde/anaconda3/envs/uc-agri-drone-env/libs"
)

add_executable(field_processor
        main/ortools_integration.cpp
        main/field_processing.cpp
        main/field_processing_binds.cpp
)

target_link_libraries(field_processor geos gdal_i)

add_subdirectory(external/pybind11)

pybind11_add_module(field_processing
        main/field_processing.cpp
        main/field_processing_binds.cpp
)

target_link_libraries(field_processing
        PRIVATE
        geos
        gdal_i
)