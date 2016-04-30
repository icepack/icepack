
#include <random>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <icepack/inverse/regularization.hpp>

using namespace dealii;
using namespace icepack;

using Fn = dealii::ScalarFunctionFromFunctionObject<2>;


template <class Regularizer>
bool test_regularizer(
  const Discretization<2>& dsc,
  const double alpha,
  const double tolerance
)
{
  const Regularizer regularizer(dsc, alpha);

  // Check that filtering a constant function gives the same function back
  {
    const Field<2> q = interpolate(dsc, dealii::ConstantFunction<2>(1.0));
    const Field<2> p = regularizer.filter(q, transpose(q));
    if (std::abs(regularizer(q)) > tolerance) return false;
    if (dist(p, q) / norm(q) > tolerance) return false;
  }

  // Check that filtering a noisy field reduces its energy
  {
    std::random_device device;
    std::mt19937 rng(device());
    std::normal_distribution<> normal_random_variable(0, 1);

    Field<2> z(dsc);
    for (auto& v: z.get_coefficients()) v = normal_random_variable(rng);

    const Field<2> p = regularizer.filter(z, transpose(z));
    if (regularizer(p) > regularizer(z)) return false;
  }

  return true;
}


int main()
{
  const SphericalManifold<2> circle(Point<2>(0.0, 0.0));
  Triangulation<2> triangulation;
  GridGenerator::hyper_ball(triangulation);
  triangulation.set_all_manifold_ids_on_boundary(0);
  triangulation.set_manifold(0, circle);
  const unsigned int num_levels = 5;
  triangulation.refine_global(num_levels);

  const Discretization<2> dsc(triangulation, 1);

  const double dx = dealii::GridTools::minimal_cell_diameter(triangulation);

  // Pick a smoothing length for the regularizers
  const double alpha = 0.125;

  if (not test_regularizer<inverse::SquareGradient<0, 2> >(dsc, alpha, dx*dx))
    return 1;

  if (not test_regularizer<inverse::TotalVariation<0, 2> >(dsc, alpha, dx*dx))
    return 1;

  return 0;
}
