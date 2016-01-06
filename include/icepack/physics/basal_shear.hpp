
#ifndef ICEPACK_BASAL_SHEAR_HPP
#define ICEPACK_BASAL_SHEAR_HPP

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

namespace icepack {

  using dealii::Tensor;
  using dealii::SymmetricTensor;

  namespace basal_shear {

    struct nonlinear
    {
      nonlinear(const double m, const double tau0, const double u0);

      double operator()(const double beta, const Tensor<1, 2>& u) const;

    private:
      const double m, tau0, u0;
    };

    struct linearized
    {
      linearized(const double m, const double tau0, const double u0);

      SymmetricTensor<2, 2>
      operator()(const double beta, const Tensor<1, 2>& u) const;

    private:
      const double m, tau0, u0;
    };

  }

}

#endif
