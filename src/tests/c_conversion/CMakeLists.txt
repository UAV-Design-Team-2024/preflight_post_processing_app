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

# Test
add_executable(test_geos_wkt
        main/test.cpp)

target_include_directories(test_geos_wkt PRIVATE
        "C:/OSGeo4W/include"
        "C:/OSGeo4W/include/geos"
)
target_link_libraries(test_geos_wkt
        geos
)

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