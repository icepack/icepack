
#include <deal.II/grid/grid_generator.h>

#include "shallow_shelf.hpp"

using namespace dealii;
using namespace ShallowShelfApproximation;

/**
 * This program tests the correctness of the our implementation of the shallow
 * shelf approximation using the method of manufactured solutions.
 * Constant ice flow and constant thickness are a simple solution of SSA.
 */


const double eps = 0.2;
const double temp = 263.15;
const double surf = 8 * viscosity(temp, eps) * eps / (rho_ice * gravity);


class BoundaryVelocity : public TensorFunction<1, 2>
{
public:
  BoundaryVelocity () : TensorFunction<1, 2>() {}
  virtual Tensor<1, 2> value (const Point<2>& x) const
  {
    Tensor<1, 2> v;
    v[0] = 100.0 + eps * x[0];
    v[1] = 0.0;
    return v;
  }
};


int main()
{
  dealii::deallog.depth_console (0);

  const Point<2> p1(0.0, 0.0), p2(2000.0, 500.0);
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);
  triangulation.refine_global(2);

  auto surface = ConstantFunction<2>(surf);
  auto bed     = ConstantFunction<2>(-2000.0);
  BoundaryVelocity boundary_velocity;

  ShallowShelf shallow_shelf(triangulation, surface, bed, boundary_velocity);

  shallow_shelf.setup_system(true);

  const auto& system_matrix = shallow_shelf.get_system_matrix();

  std::cout << system_matrix.m() << ", " << system_matrix.n() << std::endl;

  return 0;
}
