
#ifndef ICE_THICKNESS_HPP
#define ICE_THICKNESS_HPP

#include <deal.II/base/function.h>

using dealii::Function;
using dealii::Point;


/* A deal.II Function object describing the ice thickness, assuming
   known ice surface and bed elevations. */
class IceThickness : public Function<2>
{
public:
  IceThickness (const Function<2>& _surface, const Function<2>& _bed);
  virtual double value (const Point<2>& x, const unsigned int component = 0) const;
  virtual void value_list (const std::vector<Point<2> >& points,
                           std::vector<double>&          values,
                           const unsigned int            component = 0) const;

private:
  const Function<2>& surface;
  const Function<2>& bed;
};

#endif
