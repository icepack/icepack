
set(SRCS
  grid_data.cpp
  ice_thickness.cpp
  util/tensor_function_utils.cpp
  physics/viscosity.cpp
  physics/basal_shear.cpp
  discretization.cpp
  glacier_models/depth_averaged_model.cpp
  glacier_models/ice_shelf.cpp
  glacier_models/ice_stream.cpp
  inverse/error_functionals.cpp
  inverse/ice_shelf.cpp
  )

add_library(icepack ${SRCS})
DEAL_II_SETUP_TARGET(icepack)
target_link_libraries(icepack gdal)

install(TARGETS icepack DESTINATION lib)
