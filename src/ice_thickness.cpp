
#include <icepack/ice_thickness.hpp>
#include <icepack/physics/constants.hpp>

namespace icepack
{
  using dealii::StandardExceptions::ExcDimensionMismatch;
  using dealii::StandardExceptions::ExcNotImplemented;

  IceThickness::IceThickness (const Function<2>& _surface,
                              const Function<2>& _bed)
    :
    surface (_surface),
    bed (_bed)
  {}


  double IceThickness::value (const Point<2>& x,
                              const unsigned int component) const
  {
    Assert(component == 0, ExcNotImplemented());

    const double s = surface.value(x);
    const double b = bed.value(x);
    return std::min(s - b, rho_water / (rho_water - rho_ice) * s);
  }


  void IceThickness::value_list (const std::vector<Point<2> >& points,
                                 std::vector<double>&          values,
                                 const unsigned int            component) const
  {
    Assert (component == 0, ExcNotImplemented());
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int i = 0; i < n_points; ++i)
      values[i] = IceThickness::value(points[i]);
  }

}
