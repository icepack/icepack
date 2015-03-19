
#ifndef DRIVING_STRESS_HPP
#define DRIVING_STRESS_HPP

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

using dealii::Point;
using dealii::Function;
using dealii::Vector;

class DrivingStress : public Function<2>
{
public:
  DrivingStress(const Function<2>& _thickness, const Function<2>& _surface);
  void vector_value(const Point<2>& x, Vector<double>& values) const;

private:
  const Function<2>& thickness;
  const Function<2>& surface;
};

#endif
