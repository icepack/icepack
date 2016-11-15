
#include <random>
#include <icepack/inverse/regularization.hpp>
#include "../testing.hpp"

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
void test_regularizer(
  const Discretization<2>& dsc,
  const Regularizer& regularizer,
  const double tolerance
)
{
  // Check that the penalty for a constant function is 0
  const Field<2> one = interpolate(dsc, dealii::ConstantFunction<2>(1.0));
  check_real(regularizer(one), 0, tolerance);

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

  double variance = 0.0;
  for (const auto& diff: diffs) variance += (diff - mean) * (diff - mean);
  variance /= num_samples;
  const double std_dev = std::sqrt(variance);

  // TODO: This is just... laughably ad hoc, come up with a better test.
  check_real(std_dev/mean, 0.0, 5.0e-2);
}


int main()
{
  const double width = 2.0, height = 1.0;
  Triangulation<2> tria = testing::rectangular_glacier(width, height);
  const double dx = dealii::GridTools::minimal_cell_diameter(tria);

  const Discretization<2> dsc(tria, 1);

  // Pick a secondary length scale for the total variation
  const double gamma = 1.0 / width;

  const inverse::SquareGradient<2> square_gradient(dsc);
  test_regularizer(dsc, square_gradient, dx*dx);

  const inverse::TotalVariation<2> total_variation(dsc, gamma);
  test_regularizer(dsc, total_variation, dx*dx);

  // Compute the total variation of the cosh function and compare with exact
  // value of the integral
  const double b = width;
  const double a = 1.0;
  const Fn Cosh([&](const Point<2>& x){ return a * std::cosh(x[0] / b); });
  const Field<2> cosh = interpolate(dsc, Cosh);

  const double exact_tv = gamma * height * (b * std::sinh(width / b) - width);
  check_real(total_variation(cosh), exact_tv, dx);

  const double exact_square_gradient =
    std::pow(a/b, 2) * height * (b * std::sinh(2 * width / b) / 2 - width) / 4;
  check_real(square_gradient(cosh), exact_square_gradient, dx);

  return 0;
}
