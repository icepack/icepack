
#include <cmath>

#include <icepack/physics/basal_shear.hpp>

namespace icepack {

  using dealii::unit_symmetric_tensor;
  using dealii::identity_tensor;
  using dealii::outer_product;

  namespace {
    const SymmetricTensor<2, 2> I = unit_symmetric_tensor<2>();

    namespace DefaultPhysicalParams {
      const double m = 3.0, tau0 = 0.1, u0 = 100.0;
    }
  }

  BasalShear::BasalShear()
    :
    m(DefaultPhysicalParams::m),
    tau0(DefaultPhysicalParams::tau0),
    u0(DefaultPhysicalParams::u0)
  {}

  BasalShear::BasalShear(const double m, const double tau0, const double u0)
    :
    m(m),
    tau0(tau0),
    u0(u0)
  {}

  template <>
  SymmetricTensor<2, 2>
  BasalShear::K<nonlinear>(const double beta, const Tensor<1, 2>& u) const
  {
    const double U = u.norm();
    return tau0 * std::exp(beta) * std::pow(U/u0, 1/m - 1) / u0 * I;
  }

  template <>
  SymmetricTensor<2, 2>
  BasalShear::K<linearized>(const double beta, const Tensor<1, 2>& u) const
  {
    const double U = u.norm();
    const double C = tau0 * std::exp(beta) * std::pow(U/u0, 1/m - 1) / u0;
    const Tensor<1, 2> v = u/U;
    return C * (I + (1.0/m - 1) * outer_product(v, v));
  }

}
