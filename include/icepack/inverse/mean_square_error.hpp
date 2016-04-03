
#ifndef ICEPACK_INVERSE_MEAN_SQUARE_ERROR_HPP
#define ICEPACK_INVERSE_MEAN_SQUARE_ERROR_HPP

#include <icepack/field.hpp>

namespace icepack {
  namespace inverse {

    /**
     * Calculate the misfit between a modeled velocity field and observations
     * weighted by the estimated standard deviation `sigma` of the measurements
     */
    double mean_square_error(
      const VectorField<2>& u_model,
      const VectorField<2>& u_observed,
      const Field<2>& sigma
    );

  }
}

#endif
