
#include <cstring>
#include <iostream>
#include <icepack/physics/constants.hpp>
#include <icepack/ice_thickness.hpp>
#include "testing.hpp"

using namespace dealii;
using namespace icepack;

int main()
{
  ScalarFunctionFromFunctionObject<2>
    bed ([](const Point<2>& x)
         {
           return -400.0 + 0.02 * x(0);
         });

  ScalarFunctionFromFunctionObject<2>
    surface ([](const Point<2>& x)
             {
               return 200.0 - 0.04 * x(0);
             });

  IceThickness thickness (surface, bed);

  double r = rho_water / (rho_water - rho_ice);
  double L = (200 * r - 600) / (0.04 * r - 0.06);

  Point<2> x = {L - 100.0, 0.0};
  check(std::abs(thickness.value(x) - (surface.value(x) - bed.value(x))) < 1e-5);

  x(0) = L + 100.0;
  check(std::abs(thickness.value(x) - r * surface.value(x)) < 1e-5);

  return 0;
}
