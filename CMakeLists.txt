
# -------------
# Do not modify
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.2 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()
DEAL_II_INITIALIZE_CACHED_VARIABLES()
# ---------------


project(icepack)

enable_testing()

include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory("${CMAKE_SOURCE_DIR}/test")

set(LIBRARY_OUTPUT_PATH ${icepack_BINARY_DIR}/lib)

add_library(icepack
  src/read_gridded_data.cpp
  src/ice_thickness.cpp
  src/physics/viscosity.cpp
  src/numerics/linear_solve.cpp
  src/glacier_models/depth_averaged_model.cpp
  src/glacier_models/ice_shelf.cpp
  src/glacier_models/ice_stream.cpp)
DEAL_II_SETUP_TARGET(icepack)
target_link_libraries(icepack gdal)

install (TARGETS icepack DESTINATION lib)
install (DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
