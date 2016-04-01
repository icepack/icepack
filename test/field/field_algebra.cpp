
#include <deal.II/grid/grid_generator.h>

#include <icepack/field.hpp>

using dealii::Point;
using dealii::Tensor;
using dealii::Function;
using dealii::TensorFunction;
using dealii::Triangulation;
namespace GridGenerator = dealii::GridGenerator;
using dealii::ScalarFunctionFromFunctionObject;
using dealii::FEValues;
namespace FEValuesExtractors = dealii::FEValuesExtractors;

using icepack::Discretization;
using icepack::Field;
using icepack::VectorField;
using icepack::FieldType;
using icepack::interpolate;
namespace DefaultUpdateFlags = icepack::DefaultUpdateFlags;


template <int rank, int dim>
bool test_scalar_multiplication(
  const FieldType<rank, dim>& phi, const double tolerance
)
{
  FieldType<rank, dim> psi(phi);

  psi *= 2.0;

  dealii::QGauss<dim> quad = psi.get_discretization().quad();
  const unsigned int n_q_points = quad.size();
  std::vector<typename FieldType<rank, dim>::value_type>
    phi_values(n_q_points), psi_values(n_q_points);

  FEValues<dim> fe_values(psi.get_fe(), quad, DefaultUpdateFlags::flags);
  const typename FieldType<rank, dim>::extractor_type ex(0);

  for (auto cell: psi.get_dof_handler().active_cell_iterators()) {
    fe_values.reinit(cell);

    fe_values[ex].get_function_values(psi.get_coefficients(), psi_values);
    fe_values[ex].get_function_values(phi.get_coefficients(), phi_values);

    for (unsigned int q = 0; q < n_q_points; ++q) {
      // Admittedly this looks a little weird. Why not just use std::abs to
      // compute the difference between the two values instead of creating the
      // temporary variable `delta`? The reason is that this quantity has a
      // different type depending on whether we're looking at scalar or vector
      // fields; in the one case it's a `double`, in the other it's a
      // `Tensor<1, dim>`. The `auto` type declaration is hiding a rather large
      // subtlety here! Computing the square of the value is the only way to
      // get its magnitude in the same way for scalars or vectors.
      const auto delta = 2*phi_values[q] - psi_values[q];
      if (delta*delta > tolerance*tolerance) return false;
    }
  }

  return true;
}


template <int rank, int dim>
bool test_addition(
  const FieldType<rank, dim>& phi1,
  const FieldType<rank, dim>& phi2,
  const double tolerance
)
{
  FieldType<rank, dim> psi(phi1);
  psi += phi2;

  dealii::QGauss<dim> quad = psi.get_discretization().quad();
  const unsigned int n_q_points = quad.size();
  std::vector<typename FieldType<rank, dim>::value_type>
    psi_values(n_q_points), phi1_values(n_q_points), phi2_values(n_q_points);

  FEValues<dim> fe_values(psi.get_fe(), quad, DefaultUpdateFlags::flags);
  const typename FieldType<rank, dim>::extractor_type ex(0);

  for (auto cell: psi.get_dof_handler().active_cell_iterators()) {
    fe_values.reinit(cell);

    fe_values[ex].get_function_values(psi.get_coefficients(), psi_values);
    fe_values[ex].get_function_values(phi1.get_coefficients(), phi1_values);
    fe_values[ex].get_function_values(phi2.get_coefficients(), phi2_values);

    for (unsigned int q = 0; q < n_q_points; ++q) {
      const auto delta = psi_values[q] - phi1_values[q] - phi2_values[q];
      if (delta*delta > tolerance*tolerance) return false;
    }
  }

  return true;

}


template <int dim>
class F1 : public TensorFunction<1, dim>
{
public:
  F1() : TensorFunction<1, dim>() {}

  Tensor<1, dim> value(const Point<dim>& x) const
  {
    Tensor<1, dim> v;
    for (size_t k = 0; k < dim; ++k) v[k] = x[(k+1)%dim];
    return v;
  }
};


template <int dim>
class F2 : public TensorFunction<1, dim>
{
public:
  F2() : TensorFunction<1, dim>() {}

  Tensor<1, dim> value(const Point<dim>& x) const
  {
    Tensor<1, dim> v;
    for (size_t k = 0; k < dim; ++k) v[k] = -2 * x[k] * exp(-x*x);
    return v;
  }
};


int main()
{
  const unsigned int num_levels = 5;
  const double dx = 1.0 / (1 << num_levels);

  const Point<2> p1(0.0, 0.0), p2(1.0, 1.0);
  Triangulation<2> mesh;
  GridGenerator::hyper_rectangle(mesh, p1, p2);
  mesh.refine_global(num_levels);

  const Discretization<2> discretization(mesh, 1);

  const ScalarFunctionFromFunctionObject<2>
    Phi1([](const Point<2>& x){return std::exp(-x*x);}),
    Phi2([](const Point<2>& x){return x[0]*x[0] - x[1]*x[1];});

  const Field<2> phi1 = interpolate(discretization, Phi1);
  const Field<2> phi2 = interpolate(discretization, Phi2);

  if (!test_scalar_multiplication(phi1, dx*dx)) return 1;
  if (!test_addition(phi1, phi2, dx*dx)) return 1;

  const VectorField<2> f1 = interpolate(discretization, F1<2>());
  const VectorField<2> f2 = interpolate(discretization, F2<2>());

  if (!test_scalar_multiplication(f1, dx*dx)) return 1;
  if (!test_addition(f1, f2, dx*dx)) return 1;

  return 0;
}
