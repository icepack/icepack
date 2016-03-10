
#include <cmath>

#include <icepack/physics/constants.hpp>
#include <icepack/physics/viscosity.hpp>

namespace icepack {

  using dealii::unit_symmetric_tensor;
  using dealii::identity_tensor;
  using dealii::outer_product;


  /*
   * Procedures for computing the temperature- and strain rate-dependent
   * viscosity of ice.
   */

  const double transition_temperature = 263.215;
  const double A0_cold = 3.985e-13 * year_in_sec * 1.0e18; // MPa^{-3} a^{-1}
  const double A0_warm = 1.916e3   * year_in_sec * 1.0e18;
  const double Q_cold  = 60;
  const double Q_warm  = 139;

  double rate_factor(const double temperature)
  {
    const bool cold = (temperature < transition_temperature);
    const double A0 = cold ? A0_cold : A0_warm;
    const double Q  = cold ? Q_cold  : Q_warm;

    return A0 * std::exp(-Q / (ideal_gas * temperature));
  }


  Rheology::Rheology(const double n)
    :
    n(n)
  {}

  double Rheology::operator()(const double temperature) const
  {
    const double A = rate_factor(temperature);
    return std::pow(A, -1.0/n) / 2;
  }



  /*
   * Procedures for computing the constitutive tensor for a glacier model, i.e.
   * the rank-4 tensor that relates the stress tensor to the strain rate tensor
   */

  namespace {
    const SymmetricTensor<2, 2> I = unit_symmetric_tensor<2>();
    const SymmetricTensor<4, 2> II = identity_tensor<2>();
    const SymmetricTensor<4, 2> C = II + outer_product(I, I);
  }

  ConstitutiveTensor::ConstitutiveTensor(const double n)
    :
    rheology(n)
  {}

  SymmetricTensor<4, 2> ConstitutiveTensor::nonlinear(
    const double h,
    const double T,
    const SymmetricTensor<2, 2> eps
  ) const
  {
    const double n = rheology.n;
    const double tr = trace(eps);
    const double eps_e = sqrt((eps * eps + tr * tr)/2);
    const double nu = h * rheology(T) * std::pow(eps_e, -2.0/n);
    return 2 * nu * C;
  }

  SymmetricTensor<4, 2> ConstitutiveTensor::linearized(
    const double h,
    const double T,
    const SymmetricTensor<2, 2> eps
  ) const
  {
    const double n = rheology.n;
    const double tr = trace(eps);
    const double eps_e = sqrt((eps * eps + tr * tr)/2);
    const SymmetricTensor<2, 2> gamma = (eps + tr * I) / eps_e;
    const double nu = h * rheology(T) * std::pow(eps_e, -2.0/n);
    return 2 * nu * (C + (1-n)/(2*n) * outer_product(gamma, gamma));
  }

}
