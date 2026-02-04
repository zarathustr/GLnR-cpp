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

add_executable(glnr_fixed_corr_stats
  src/glnr_fixed_corr_stats.cpp
)

add_executable(glnr_monte_carlo_cov
  src/glnr_monte_carlo_cov.cpp
)

foreach(_tgt glnr_register glnr_fixed_corr_stats glnr_monte_carlo_cov)
  target_include_directories(${_tgt}
    PRIVATE
      ${PCL_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  # NOTE: Some PCL configs (esp. older MacPorts builds) put non-"-D" compiler
  # flags (e.g. "-ffloat-store") into PCL_DEFINITIONS. Passing those via
  # target_compile_definitions(...) will incorrectly prepend "-D" and break the
  # build. Treat PCL_DEFINITIONS as compile options instead.
  link_directories(${PCL_LIBRARY_DIRS})
  target_compile_options(${_tgt} PRIVATE -w -O3)
  target_link_libraries(${_tgt} PRIVATE ${PCL_LIBRARIES})
endforeach()

# If you build PCL with VTK enabled, VTK libs are already propagated
# through PCL_LIBRARIES. Otherwise, .vtk loading will be disabled at runtime.

message(STATUS "PCL version: ${PCL_VERSION}")
