
#ifndef ICEPACK_VISCOSITY_HPP
#define ICEPACK_VISCOSITY_HPP

#include <deal.II/base/symmetric_tensor.h>

namespace icepack {

  using dealii::SymmetricTensor;

  double rate_factor(const double temperature);
  double viscosity(const double temperature, const double strain_rate);

  namespace SSA {
    SymmetricTensor<4, 2> nonlinear(
      const double temperature,
      const double thickness,
      const SymmetricTensor<2, 2> strain_rate
    );

    SymmetricTensor<4, 2> linearized(
      const double temperature,
      const double thickness,
      const SymmetricTensor<2, 2> strain_rate
    );
  }

}

#endif
