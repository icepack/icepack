
# -------------
# Do not modify
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)

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
# --------------


project(square_ice_shelf)

set(TARGET
  square_ice_shelf)

set(SOURCES
  square_ice_shelf.cpp)

include_directories(${PROJECT_SOURCE_DIR})
add_executable(${TARGET} ${SOURCES})
DEAL_II_SETUP_TARGET(${TARGET})
target_link_libraries(${TARGET} icepack)
