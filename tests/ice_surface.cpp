
#include "physical_constants.hpp"
#include "ice_surface.hpp"

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
    thickness ([](const Point<2>& x)
               {
                 return 600.0 - 0.04 * x(0);
               });

  IceSurface surface (bed, thickness);

  double r = rho_ice / rho_water;
  double L = (200 - (1-r) * 600) / (0.02 + (1-r) * 0.04);

  const Point<2> x = {L - 100.0, 0.0};

  if (std::abs(surface.value(x) - bed.value(x) - thickness.value(x)) > 1e-5)
    {
      std::cout << "Wrong value for ice surface elevation"
                << "before grounding line!" << std::endl;
      return 1;
    }

  const Point<2> y = {L + 100.0, 0.0};

  if (std::abs(surface.value(y) - (1-r) * thickness.value(y)) > 1e-5)
    {
      std::cout << "Wrong value for ice surface elevation"
                << "after grounding line!" << std::endl;
    }

  return 0;
}
