
# -------------
# Do not modify
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.4 QUIET
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

find_package(GDAL)
include_directories(${GDAL_INCLUDE_DIRS})

set(LIBRARY_OUTPUT_PATH ${icepack_BINARY_DIR}/lib)

include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory("${CMAKE_SOURCE_DIR}/src")
add_subdirectory("${CMAKE_SOURCE_DIR}/test")

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
