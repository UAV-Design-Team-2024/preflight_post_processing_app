cmake_minimum_required(VERSION 3.24)
project(field_converter)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include paths
include_directories(
        "C:/Users/corde/anaconda3/envs/uc-agri-drone-env/include"
        external/eigen
        external/pybind11/include/
        external/or-tools/include/
        external/OSGeo4W/include/
        main
)

# Link directories (including OR-Tools build output)
link_directories(
        "C:/Users/corde/anaconda3/envs/uc-agri-drone-env/libs"
        external/or-tools/lib/
        external/OSGeo4W/lib/

)

# Test executable
add_executable(test_geos_wkt
        main/test.cpp
)

target_include_directories(test_geos_wkt PRIVATE
        external/OSGeo4W/include/
)

target_link_libraries(test_geos_wkt
        geos_c
)

# Pybind11 module
add_subdirectory(external/pybind11)

pybind11_add_module(field_processing_c_module
        main/field_processing.cpp
        main/ortools_integration.cpp
        main/field_processing_binds.cpp
)

target_link_libraries(field_processing_c_module
        PRIVATE
        geos_c
        gdal_i
        ortools_full
        libprotobuf
        zlib
        utf8_range
        utf8_validity
)
