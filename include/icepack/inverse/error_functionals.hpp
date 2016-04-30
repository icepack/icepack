
#ifndef ICEPACK_INVERSE_MEAN_SQUARE_ERROR_HPP
#define ICEPACK_INVERSE_MEAN_SQUARE_ERROR_HPP

#include <icepack/field.hpp>

namespace icepack {
  /**
   * This namespace contains procedures for the solution of inverse problems
   * for e.g. basal shear stress and ice temperature, which are not observable
   * directly through remote sensing methods.
   */
  namespace inverse {

    /**
     * Calculate the mean-square difference between a modeled velocity field
     * and observations, weighted by the estimated standard deviation `sigma`
     * of the measurements
     */
    double square_error(
      const VectorField<2>& u_model,
      const VectorField<2>& u_observed,
      const Field<2>& sigma
    );


    /**
     * Calculate the difference between the modeled and observed velocities,
     * weighted by the standard deviation of the measurements
     */
    DualVectorField<2> misfit(
      const VectorField<2>& u_model,
      const VectorField<2>& u_observed,
      const Field<2>& sigma
    );

  }
}

#endif
