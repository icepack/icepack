
#include "physical_constants.hpp"
#include "ice_thickness.hpp"
#include "driving_stress.hpp"

using namespace dealii;


class SurfaceElevation : public Function<2>
{
  double value (const Point<2>& x, const unsigned int component = 0) const
  {
    return 200.0 - 0.04 * x(0);
  }

  Tensor<1, 2> gradient (const Point<2>& x, const unsigned int component = 0) const
  {
    Tensor<1, 2> grad;
    grad[0] = -0.04;
    grad[1] =  0.0;
    return grad;
  }
};


int main (int argc, char **argv)
{

  ScalarFunctionFromFunctionObject<2>
    bed ([](const Point<2>& x)
         {
           return -400.0 + 0.02 * x(0);
         });

  SurfaceElevation surface;

  double r = rho_water / (rho_water - rho_ice);
  double L = (200 * r - 600) / (0.04 * r - 0.06);

  IceThickness thickness(bed, surface);
  DrivingStress driving_stress (thickness, surface);

  Point<2> x = {L - 100.0, 0.0};
  Vector<double> tau_d(2);

  driving_stress.vector_value(x, tau_d);
  if (std::abs(tau_d[0] - rho_ice * gravity * thickness.value(x) * 0.04) > 1e-5
      or std::abs(tau_d[1]) > 1e-5)
  {

    std::cout << tau_d[0] << ", " << tau_d[1] << std::endl;
    std::cout << rho_ice * gravity * 0.04 * thickness.value(x) << ", " << 0.0 << std::endl;

    std::cout << "Wrong value for driving stress "
              << "before grounding line!" << std::endl;
    return 1;
  }

  x(0) = L + 100.0;

  if (std::abs(tau_d[0] - rho_ice * gravity * thickness.value(x) * 0.04) > 1e-5
      or std::abs(tau_d[1]) > 1e-5)
  {
    std::cout << tau_d[0] << ", " << tau_d[1] << std::endl;
    std::cout << rho_ice * gravity * 0.04 * thickness.value(x) << ", " << 0.0 << std::endl;

    std::cout << "Wrong value for driving stress "
              << "after grounding line!" << std::endl;
    return 1;
  }

  return 0;
}
