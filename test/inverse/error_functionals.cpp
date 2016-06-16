
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

  Assert(abs(inverse::square_error(u_model, u_true, sigma) - error) < dx*dx,
         ExcInternalError());

  return 0;
}
