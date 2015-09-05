
#include <cstring>
#include <iostream>

#include "physical_constants.hpp"
#include "ice_thickness.hpp"

using namespace dealii;

int main (int argc, char **argv)
{
  bool verbose = false;
  if (strcmp(argv[argc-1], "-v") == 0) verbose = true;

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

  if (std::abs(thickness.value(x) - (surface.value(x) - bed.value(x))) > 1e-5)
  {
    std::cout << "Wrong value for ice thickness "
              << "before grounding line!" << std::endl;
    return 1;
  }

  x(0) = L + 100.0;

  if (std::abs(thickness.value(x) - r * surface.value(x)) > 1e-5)
  {
    std::cout << "Wrong value for ice thickness "
              << "after grounding line!" << std::endl;
    return 1;
  }


  if (verbose) std::cout << "Done checking ice thickness!" << std::endl;

  return 0;
}
