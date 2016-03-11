
#ifndef ICEPACK_VISCOSITY_HPP
#define ICEPACK_VISCOSITY_HPP

#include <deal.II/base/symmetric_tensor.h>

#include <icepack/physics/linearity.hpp>

namespace icepack {

  using dealii::SymmetricTensor;

  double rate_factor(const double temperature);


  struct Rheology
  {
    Rheology(const double n);

    double
    operator()(const double temperature) const;

    const double n;
  };


  /**
   * This class is for calculating the rank-4 tensor describing the relation
   * between the ice strain rate and the membrane stress tensor.
   */
  struct ConstitutiveTensor
  {
    /**
     * Create a constitutive tensor object for a given value of the ice
     * rheology exponent `n` in Glen's flow law
     */
    ConstitutiveTensor(const double n);

    /**
     * Calculate the consitutive tensor including the nonlinear dependence on
     * the ice strain rate.
     */
    template <Linearity linearity>
    SymmetricTensor<4, 2> C(
      const double thickness,
      const double temperature,
      const SymmetricTensor<2, 2> strain_rate
    ) const;

    /**
     * Parameterization for how the ice rheology depends on temperature
     */
    const Rheology rheology;
  };

}

#endif
