
#include "physical_constants.hpp"
#include "ice_thickness.hpp"
#include "driving_stress.hpp"

using namespace dealii;

int main (int argc, char **argv)
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

  double r = rho_water / (rho_water - rho_ice);
  double L = (200 * r - 600) / (0.04 * r - 0.06);

  IceThickness thickness(bed, surface);
  DrivingStress driving_stress (thickness, surface);

  Point<2> x = {L - 100.0, 0.0};
  Vector<double> tau_d(2);

  driving_stress.vector_value(x, tau_d);
  if (std::abs(tau_d[0] - 0.04 * thickness.value(x)) > 1e-5
      or std::abs(tau_d[1]) > 1e-5)
  {
    std::cout << "Wrong value for driving stress "
              << "before grounding line!" << std::endl;
    return 1;
  }

  x(0) = L + 100.0;

  if (std::abs(tau_d[0] - 0.04 * thickness.value(x)) > 1e-5
      or std::abs(tau_d[1]) > 1e-5)
  {
    std::cout << "Wrong value for driving stress "
              << "after grounding line!" << std::endl;
    return 1;
  }

  return 0;
}
