
#include <random>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <icepack/inverse/regularization.hpp>

using namespace dealii;
using namespace icepack;

using Fn = dealii::ScalarFunctionFromFunctionObject<2>;

int main()
{
  const SphericalManifold<2> circle(Point<2>(0.0, 0.0));
  Triangulation<2> triangulation;
  GridGenerator::hyper_ball(triangulation);
  triangulation.set_all_manifold_ids_on_boundary(0);
  triangulation.set_manifold(0, circle);
  const unsigned int num_levels = 5;
  triangulation.refine_global(num_levels);

  const double dx = dealii::GridTools::minimal_cell_diameter(triangulation);

  const Discretization<2> dsc(triangulation, 1);

  // Create a regularization object for low-pass filtering
  const double alpha = 0.125;
  inverse::SquareGradient<0, 2> regularizer(dsc, alpha);

  // Check that filtering a constant function gives the same function back
  {
    const Field<2> q = interpolate(dsc, dealii::ConstantFunction<2>(1.0));
    Assert(std::abs(regularizer(q)) < dx*dx, ExcInternalError());

    const Field<2> p = regularizer.filter(q, transpose(q));
    Assert(dist(p, q) / norm(q) < dx*dx, ExcInternalError());
  }

  // Check that filtering an arbitrary function reduces its energy
  {
    std::random_device device;
    std::mt19937 rng(device());
    std::normal_distribution<> normal_random_variable(0, 1);

    Field<2> z(dsc);
    for (auto& v: z.get_coefficients()) v = normal_random_variable(rng);

    const Field<2> p = regularizer.filter(z, transpose(z));
    Assert(regularizer(p) < regularizer(z), ExcInternalError());
  }

  return 0;
}
