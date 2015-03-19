
#ifndef ICE_SURFACE_HPP
#define ICE_SURFACE_HPP

#include <deal.II/base/function.h>

using dealii::Function;
using dealii::Point;


/* Given the bedrock elevation and ice thickness, computes the ice surface
   elevation. */
class IceSurface : public Function<2>
{
public:
  IceSurface(const Function<2>& _bed, const Function<2>& _thickness);
  double value(const Point<2>& x, const unsigned int component) const;

private:
  const Function<2>& bed;
  const Function<2>& thickness;
};


#endif
