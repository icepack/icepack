
#include <cmath>

#include "icepack/physical_constants.hpp"

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

double viscosity(const double temperature, const double strain_rate)
{
  const double A = rate_factor(temperature);
  return std::pow(A * strain_rate * strain_rate, -1.0/3) / 2;
}
