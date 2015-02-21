
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

PROJECT(examples)

include_directories("${PROJECT_SOURCE_DIR}/include")
set(LIBRARY_OUTPUT_PATH ${examples_BINARY_DIR}/lib)

ADD_LIBRARY(stuff src/read_gridded_data.cpp)
DEAL_II_SETUP_TARGET(stuff)

ADD_EXECUTABLE(read_qgis tests/read_qgis.cpp)
DEAL_II_SETUP_TARGET(read_qgis)
TARGET_LINK_LIBRARIES(read_qgis stuff)

#ADD_EXECUTABLE(example-01 example-01.cc)
#DEAL_II_SETUP_TARGET(example-01)

#ADD_EXECUTABLE(example-02 example-02.cc)
#DEAL_II_SETUP_TARGET(example-02)

#ADD_EXECUTABLE(example-03 example-03.cc)
#DEAL_II_SETUP_TARGET(example-03)