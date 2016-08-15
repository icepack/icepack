
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

# ----------------
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


# ---------------
#FIND_PACKAGE(icepack QUIET
#  HINTS ${ICEPACK_DIR} $ENV{ICEPACK_DIR}
#  )
#if(NOT ${icepack_FOUND})
#  MESSAGE(FATAL_ERROR "\n"
#    "*** Could not locate icepack. ***\n\n"
#    "You may want to either pass a flag -DICEPACK_DIR=/path/to/icepack to cmake\n"
#    "or set an environment variable \"ICEPACK_DIR\" that contains this path."
#    )
#ENDIF()
# --------------


add_executable(ross ross.cpp)
DEAL_II_SETUP_TARGET(ross)
target_link_libraries(ross icepack gdal)
