
#ifndef PHYSICAL_CONSTANTS_HPP
#define PHYSICAL_CONSTANTS_HPP

// All units are megapascals / meters / years
constexpr double year_in_sec = 365.25 * 24 * 3600;
constexpr double gravity  = 9.81 * year_in_sec * year_in_sec;  //  m / a^2
constexpr double idealgas = 8.3144621e-3;                      //  kJ / mole * K
constexpr double Temp     = 263.15;                            //  K
constexpr double A0_cold  = 3.985e-13 * year_in_sec * 1.0e18;  //  a^{-1} MPa^{-3}
constexpr double A0_warm  = 1.916e3   * year_in_sec * 1.0e18;
constexpr double Q_cold   = 60;                                //  kJ / mole
constexpr double Q_warm   = 139;
constexpr double rho_ice  = 917 / year_in_sec * year_in_sec * 1.0e-6;
constexpr double rho_water = 1024 / year_in_sec * year_in_sec * 1.0e-6;

#endif
