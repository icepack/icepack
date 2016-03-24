
#include <icepack/discretization.hpp>

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

  // Explicitly instantiate some templates that we'll need in thecore library
  // code anyway
  template class FieldDiscretization<0, 2>;
  template class FieldDiscretization<1, 2>;
  template class Discretization<2>;
}
