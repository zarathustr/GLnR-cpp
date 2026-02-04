cmake_minimum_required(VERSION 3.5)
project(glnr_pcl_registration CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# PCL 1.8.0 ~ 1.12.0
find_package(PCL REQUIRED COMPONENTS common io filters search registration visualization)

add_executable(glnr_register
  src/glnr_register.cpp
)

target_include_directories(glnr_register
  PRIVATE
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# PCL's exported variables may include both -D... and compiler flags (e.g., -ffloat-store).
# Use compile options (not compile definitions) to avoid misinterpreting flags as macros.
link_directories(${PCL_LIBRARY_DIRS})
target_compile_options(glnr_register PRIVATE -O3 -w)


target_link_libraries(glnr_register PRIVATE ${PCL_LIBRARIES})

# If you build PCL with VTK enabled, VTK libs are already propagated
# through PCL_LIBRARIES. Otherwise, .vtk loading will be disabled at runtime.

message(STATUS "PCL version: ${PCL_VERSION}")