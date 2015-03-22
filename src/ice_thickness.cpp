
#include "ice_thickness.hpp"
#include "physical_constants.hpp"


IceThickness::IceThickness (const Function<2>& _bed,
                            const Function<2>& _surface)
  :
  bed (_bed),
  surface (_surface)
{}


double IceThickness::value (const Point<2>& x,
                            const unsigned int component) const
{
  const double s = surface.value(x);
  const double b = bed.value(x);
  return std::min(s - b, rho_water / (rho_water - rho_ice) * s);
}
