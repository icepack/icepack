
#ifndef PHYSICAL_CONSTANTS_HPP
#define PHYSICAL_CONSTANTS_HPP

namespace icepack
{
  // All units are megapascals / meters / years
  constexpr double year_in_sec = 365.25 * 24 * 3600;
  constexpr double gravity  = 9.81 * year_in_sec * year_in_sec;  //  m/a^2
  constexpr double ideal_gas = 8.3144621e-3;                     //  kJ/mole K
  constexpr double rho_ice  = 917 / (year_in_sec * year_in_sec) * 1.0e-6;
  constexpr double rho_water = 1024 / (year_in_sec * year_in_sec) * 1.0e-6;

  double rate_factor(const double temperature);
  double viscosity(const double temperature, const double strain_rate);
}

#endif
