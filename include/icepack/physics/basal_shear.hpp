
#ifndef ICEPACK_BASAL_SHEAR_HPP
#define ICEPACK_BASAL_SHEAR_HPP

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

#include <icepack/physics/linearity.hpp>

namespace icepack {

  using dealii::Tensor;
  using dealii::SymmetricTensor;

  /**
   * Function object for computing the basal friction
   */
  struct BasalShear
  {
    /**
     * Construct a basal friction function object for a given sliding rheology,
     * yield stress and yield speed
     */
    BasalShear(const double m, const double tau0, const double u0);

    /**
     * Compute the basal friction coefficient or its linearization
     */
    template <Linearity linearity>
    SymmetricTensor<2, 2> K(const double beta, const Tensor<1, 2>& u) const;

    /**
     * Sliding rheology exponent
     */
    const double m;

    /**
     * Plastic yield stress
     */
    const double tau0;

    /**
     * Yield speed
     */
    const double u0;
  };

}

#endif
