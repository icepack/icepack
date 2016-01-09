
#include <icepack/glacier_models/pde_skeleton.hpp>

namespace icepack {
  namespace DefaultUpdateFlags {

    using dealii::update_values;
    using dealii::update_gradients;
    using dealii::update_quadrature_points;
    using dealii::update_JxW_values;
    using dealii::update_normal_vectors;

    const UpdateFlags flags =
      update_values            | update_gradients |
      update_quadrature_points | update_JxW_values;

    const UpdateFlags face_flags =
      update_values         | update_quadrature_points |
      update_normal_vectors | update_JxW_values;
  }

  // Explicitly instantiate the PDE skeleton template classes for dimension 2,
  // which we'll need in the core library code anyway.
  template class PDESkeleton<2, FE_Q<2> >;
  template class PDESkeleton<2, FESystem<2> >;
}
