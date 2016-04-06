
#include <iostream>

#include <deal.II/grid/grid_generator.h>

#include <icepack/field.hpp>


const unsigned int num_levels = 6;
const double dx = 1.0/(1 << num_levels);


using namespace dealii;
using namespace icepack;
using std::abs;

const double pi = 4.0 * std::atan(1.0);

template <int dim>
class Phi1 : public Function<dim>
{
  double value(const Point<dim>& x, const unsigned int = 0) const
  {
    const Point<dim> k1(pi * 2, pi * 3);
    return std::sin(k1 * x);
  }
};


template <int dim>
class Phi2 : public Function<dim>
{
  double value(const Point<dim>& x, const unsigned int = 0) const
  {
    const Point<dim> k2(pi * 5, pi);
    return std::cos(k2 * x);
  }
};


int main()
{
  const Point<2> p1(0.0, 0.0), p2(1.0, 1.0);
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);
  triangulation.refine_global(num_levels);

  const Discretization<2> discretization(triangulation, 1);

  const Field<2> phi1 = interpolate(discretization, Phi1<2>());
  const Field<2> phi2 = interpolate(discretization, Phi2<2>());

  const double exact_distance = 1.0;

  if (abs(dist(phi1, phi2) - exact_distance) > dx) return 1;

  const double exact_inner_product = 0.0;

  if (abs(inner_product(phi1, phi2) - exact_inner_product) > dx) return 1;

  return 0;
}
