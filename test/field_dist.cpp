
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


template <int dim>
bool test_field_dist(
  const Triangulation<dim>& triangulation,
  const Function<dim>& phi1,
  const Function<dim>& phi2,
  const double exact_distance
)
{
  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const Field<dim> psi1 = interpolate(triangulation, fe, dof_handler, phi1);
  const Field<dim> psi2 = interpolate(triangulation, fe, dof_handler, phi2);

  Assert(abs(dist(psi1, psi2) - exact_distance) < dx, ExcInternalError());

  return true;
}


int main()
{
  const Point<2> p1(0.0, 0.0), p2(1.0, 1.0);
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);
  triangulation.refine_global(num_levels);

  Phi1<2> phi1;
  Phi2<2> phi2;

  const double exact_distance = 1.0;

  if (!test_field_dist(triangulation, phi1, phi2, exact_distance)) return 1;

  return 0;
}
