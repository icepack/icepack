
#include <iostream>

#include <deal.II/grid/grid_generator.h>

#include <icepack/field.hpp>


using namespace dealii;
using namespace icepack;


class Phi : public Function<2>
{
  double value(const Point<2>& x, const unsigned int = 0) const
  {
    return x[0] * x[1];
  }
};


int main()
{
  const Point<2> p1(0.0, 0.0), p2(1.0, 1.0);
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);

  FE_Q<2> finite_element(1);

  Phi phi;
  Field<2> psi = interpolate(triangulation, finite_element, phi);

  return 0;
}
