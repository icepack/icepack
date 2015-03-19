
#include "ice_surface.hpp"
#include "physical_constants.hpp"


IceSurface::IceSurface(const Function<2>& _bed, const Function<2>& _thickness)
  :
  bed (_bed),
  thickness (_thickness)
{ }


double IceSurface::value(const Point<2>& x, const unsigned int component) const
{
  const double h = thickness.value(x, 0);
  const double b = bed.value(x, 0);
  return std::max(b + h, (1.0 - rho_ice/rho_water) * h);
}
