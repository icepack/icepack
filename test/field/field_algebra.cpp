
#include <deal.II/grid/grid_generator.h>

#include <icepack/field/field.hpp>
#include <icepack/field/field_algebra.hpp>

using dealii::Point;
using dealii::Function;
using dealii::Triangulation;
namespace GridGenerator = dealii::GridGenerator;
using dealii::ScalarFunctionFromFunctionObject;
using dealii::FEValues;
namespace FEValuesExtractors = dealii::FEValuesExtractors;

using icepack::Discretization;
using icepack::Field;
using icepack::VectorField;
using icepack::interpolate;
namespace DefaultUpdateFlags = icepack::DefaultUpdateFlags;


template <int dim>
bool test_scalar_multiplication(const Field<dim>& phi, const double tolerance)
{
  Field<dim> psi;
  psi.copy_from(phi);

  psi *= 2.0;

  dealii::QGauss<dim> quad = psi.get_discretization().quad();
  const unsigned int n_q_points = quad.size();
  std::vector<double> phi_values(n_q_points), psi_values(n_q_points);

  FEValues<dim> fe_values(psi.get_fe(), quad, DefaultUpdateFlags::flags);
  const FEValuesExtractors::Scalar ex(0);

  for (auto cell: psi.get_dof_handler().active_cell_iterators()) {
    fe_values.reinit(cell);

    fe_values[ex].get_function_values(psi.get_coefficients(), psi_values);
    fe_values[ex].get_function_values(phi.get_coefficients(), phi_values);

    for (unsigned int q = 0; q < n_q_points; ++q)
      if (abs(2*phi_values[q] - psi_values[q]) > tolerance) return false;
  }

  return true;
}


template <int dim>
bool test_addition(
  const Field<dim>& phi1, const Field<dim>& phi2, const double tolerance
)
{
  Field<dim> psi;
  psi.copy_from(phi1);

  psi += phi2;

  dealii::QGauss<dim> quad = psi.get_discretization().quad();
  const unsigned int n_q_points = quad.size();
  std::vector<double>
    psi_values(n_q_points), phi1_values(n_q_points), phi2_values(n_q_points);

  FEValues<dim> fe_values(psi.get_fe(), quad, DefaultUpdateFlags::flags);
  const FEValuesExtractors::Scalar ex(0);

  for (auto cell: psi.get_dof_handler().active_cell_iterators()) {
    fe_values.reinit(cell);

    fe_values[ex].get_function_values(psi.get_coefficients(), psi_values);
    fe_values[ex].get_function_values(phi1.get_coefficients(), phi1_values);
    fe_values[ex].get_function_values(phi2.get_coefficients(), phi2_values);

    for (unsigned int q = 0; q < n_q_points; ++q)
      if (abs(psi_values[q] - phi1_values[q] - phi2_values[q]) > tolerance)
        return false;
  }

  return true;

}


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

  return 0;
}
