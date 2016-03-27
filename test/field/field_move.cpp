
/**
 * This test shows that we can return FieldType objects from functions even
 * the copy constructor for FieldType is deleted. This is possible through the
 * use of move constructors and move assignment operators.
 * This program exists more to see if it compiles or not; it doesn't throw any
 * errors in case of a failure of run-time logic.
 */

#include <iostream>

#include <deal.II/grid/grid_generator.h>

#include <icepack/field.hpp>


const unsigned int num_levels = 4;


using namespace dealii;
using namespace icepack;


template <int dim>
class Z : public Function<dim>
{
public:
  Z() {}

  double value(const Point<dim>& x, const unsigned int = 0) const
  {
    const double r = x.norm();
    return std::exp(-0.5*r*r);
  }
};


template <int dim>
class P : public Function<dim>
{
public:
  P() {}

  double value(const Point<dim>& x, const unsigned int = 0) const
  {
    const double r = x.norm();
    return 1 - r*r;
  }
};


template <int dim>
Field<dim> gaussian(const Discretization<dim>& discretization)
{
  return interpolate(discretization, Z<dim>());
}


template <int dim>
Field<dim> parabola(const Discretization<dim>& discretization)
{
  return interpolate(discretization, P<dim>());
}


int main()
{
  const Point<2> p1(-1.0, -1.0), p2(1.0, 1.0);
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);
  triangulation.refine_global(num_levels);

  const Discretization<2> discretization(triangulation, 1);

  /**
   * Initialize a FieldType object with the return value from a function; this
   * utilizes the move constructor for FieldType.
   */
  Field<2> u = gaussian(discretization);

  std::cout << norm(u) << std::endl;

  /**
   * Reassign the FieldType object with the return value from another function;
   * this uses the move assignment operator.
   */
  u = parabola(discretization);

  std::cout << norm(u) << std::endl;

  return 0;
}
