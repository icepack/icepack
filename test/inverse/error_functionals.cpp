
#include <icepack/field.hpp>
#include <icepack/inverse/error_functionals.hpp>
#include "../testing.hpp"

using namespace dealii;
using namespace icepack;
using std::abs;

class TrueVelocity : public TensorFunction<1, 2>
{
public:
  Tensor<1, 2> value(const Point<2>&) const
  {
    Tensor<1, 2> v;
    v[0] = 1.0; v[1] = 0.0;
    return v;
  }
} U_true;

class ModelVelocity : public TensorFunction<1, 2>
{
public:
  Tensor<1, 2> value(const Point<2>& x) const
  {
    Tensor<1, 2> v = U_true.value(x);
    v[1] = 0.5;
    return v;
  }
} U_model;

int main()
{
  Triangulation<2> tria = testing::rectangular_glacier(1.0, 1.0);
  const double dx = dealii::GridTools::minimal_cell_diameter(tria);

  const Discretization<2> discretization(tria, 1);

  const VectorField<2>
    u_true = interpolate(discretization, U_true),
    u_model = interpolate(discretization, U_model);

  const Field<2> sigma = interpolate(discretization, ConstantFunction<2>(1.0));

  const double error = 0.125;

  check_real(inverse::square_error(u_model, u_true, sigma), error, dx*dx);


  const auto F =
    [&](const VectorField<2>& u)
    {
      return inverse::square_error(u, u_true, sigma);
    };

  const DualVectorField<2> df = inverse::misfit(u_model, u_true, sigma);
  const VectorField<2> p = -transpose(df);
  const double df_times_p = icepack::inner_product(df, p);

  const double beta_max = 1.0;

  const double cost = F(u_model);
  const double delta = 0.5;

  const size_t num_samples = 16;
  std::vector<double> errors(num_samples);
  for (size_t n = 0; n < num_samples; ++n) {
    const double delta_u = std::pow(delta, n) * beta_max;
    const double delta_F = (F(u_model + delta_u * p) - cost) / delta_u;
    errors[n] = std::abs(1.0 - delta_F / df_times_p);
  }

  check(icepack::testing::is_decreasing(errors));

  return 0;
}
