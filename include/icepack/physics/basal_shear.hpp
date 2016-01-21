
#ifndef ICEPACK_BASAL_SHEAR_HPP
#define ICEPACK_BASAL_SHEAR_HPP

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

namespace icepack {

  using dealii::Tensor;
  using dealii::SymmetricTensor;

  struct BasalShear
  {
    BasalShear(const double m, const double tau0, const double u0);

    double nonlinear(const double beta, const Tensor<1, 2>& u) const;

    SymmetricTensor<2, 2>
    linearized(const double beta, const Tensor<1, 2>& u) const;

    const double m, tau0, u0;
  };

}

#endif
