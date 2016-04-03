
#include <deal.II/grid/grid_generator.h>

#include <icepack/field.hpp>
#include <icepack/inverse/mean_square_error.hpp>

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
  const Point<2> p1(0.0, 0.0), p2(1.0, 1.0);
  const unsigned int num_levels = 5;
  const double dx = 1.0 / (1 << num_levels);
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);
  triangulation.refine_global(num_levels);

  const Discretization<2> discretization(triangulation, 1);

  const VectorField<2>
    u_true = interpolate(discretization, U_true),
    u_model = interpolate(discretization, U_model);

  const Field<2> sigma = interpolate(discretization, ConstantFunction<2>(1.0));

  const double mse = 0.125;

  Assert(abs(inverse::mean_square_error(u_model, u_true, sigma) - mse) < dx*dx,
         ExcInternalError());

  return 0;
}
