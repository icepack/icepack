
#include <random>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <icepack/inverse/regularization.hpp>

using namespace dealii;
using namespace icepack;

using Fn = dealii::ScalarFunctionFromFunctionObject<2>;


template <int dim>
Field<dim>
random_trigonometric_polynomial(
  const Discretization<dim>& dsc,
  const unsigned int degree
)
{
  std::random_device device;
  std::mt19937 rng(device());
  std::normal_distribution<> v;

  std::vector<Point<dim> > ks(degree);
  std::vector<double> as(degree);
  for (size_t n = 0; n < degree; ++n) {
    for (size_t i = 0; i < dim; ++i)
      ks[n][i] = v(rng);
    as[n] = v(rng) / (1 + ks[n].norm_square());
  }

  const auto u = [&](const Point<dim>& x)
    {
      double z = 0.0;
      for (size_t n = 0; n < degree; ++n)
        z += as[n] * cos(ks[n] * x);

      return z;
    };

  return interpolate(dsc, Fn(u));
}


template <class Regularizer>
bool test_regularizer(
  const Discretization<2>& dsc,
  const Regularizer& regularizer,
  const double tolerance
)
{
  // Check that the penalty for a constant function is 0
  const Field<2> one = interpolate(dsc, dealii::ConstantFunction<2>(1.0));
  if (std::abs(regularizer(one)) > tolerance) return false;

  // Check that computing the derivative of the functional works right
  const unsigned int degree = 5;
  const Field<2> u = random_trigonometric_polynomial<2>(dsc, degree);
  const Field<2> v = random_trigonometric_polynomial<2>(dsc, degree);

  const DualField<2> p = regularizer.derivative(u);

  const unsigned int num_samples = 10;
  std::vector<double> diffs(num_samples);
  for (size_t n = 0; n < num_samples; ++n) {
    const double delta = 1.0 / (1 << n);
    const double diff = (regularizer(u + delta*v) - regularizer(u)) / delta;
    diffs[n] = (diff - inner_product(p, v)) / delta;
  }

  double mean = 0.0;
  for (const auto& diff: diffs) mean += diff;
  mean /= num_samples;

  double std_dev = 0.0;
  for (const auto& diff: diffs) std_dev += (diff - mean) * (diff - mean);
  std_dev /= num_samples;

  if (std::abs(std_dev - std::abs(mean)) > 1.0e-2) return false;

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

  const inverse::SquareGradient<2> square_gradient(dsc, alpha);
  if (not test_regularizer(dsc, square_gradient, dx*dx))
    return 1;

  const inverse::TotalVariation<2> total_variation(dsc, alpha, 0.5);
  if (not test_regularizer(dsc, total_variation, dx*dx))
    return 1;

  return 0;
}
