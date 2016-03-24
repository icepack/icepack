
#include <iostream>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <icepack/field.hpp>


const unsigned int num_levels = 3;
const double dx = 1.0/(1 << num_levels);


using namespace dealii;
using namespace icepack;
using std::abs;

template <int dim>
class Phi : public Function<dim>
{
  double value(const Point<dim>& x, const unsigned int = 0) const
  {
    return x[0] * x[1];
  }
};


template <int dim>
class F : public TensorFunction<1, dim>
{
  Tensor<1, dim> value(const Point<dim>& x) const
  {
    Tensor<1, dim> v;
    for (unsigned int i = 0; i < dim; ++i) v[i] = x[(i + 1) % dim];
    return v;
  }
};


template <int dim>
bool
test_field(const Discretization<dim>& discretization, const Function<dim>& phi)
{
  Field<dim> psi = interpolate(discretization, phi);

  Quadrature<dim> quad = discretization.quad();
  const unsigned int n_q_points = quad.size();
  std::vector<double> phi_values(n_q_points), psi_values(n_q_points);

  FEValues<dim>
    fe_values(psi.get_fe(), quad, update_values | update_quadrature_points);
  const FEValuesExtractors::Scalar ex(0);

  for (auto cell: psi.get_dof_handler().active_cell_iterators()) {
    fe_values.reinit(cell);

    phi.value_list(fe_values.get_quadrature_points(), phi_values);
    fe_values[ex].get_function_values(psi.get_coefficients(), psi_values);

    for (unsigned int q = 0; q < n_q_points; ++q)
      Assert(abs(phi_values[q] - psi_values[q]) < 1.0e-6, ExcInternalError());
  }

  const double n = norm(psi);
  const double exact_integral = 1.0/3;
  Assert(abs(n - exact_integral) < dx*dx, ExcInternalError());

  return true;
}


template <int dim>
bool test_vector_field(
  const Discretization<dim>& discretization,
  const TensorFunction<1, dim>& f
)
{
  VectorField<dim> g = interpolate(discretization, f);

  Quadrature<dim> quad = discretization.quad();
  const unsigned int n_q_points = quad.size();

  std::vector<Tensor<1, dim>> f_values(n_q_points), g_values(n_q_points);
  FEValues<dim>
    fe_values(g.get_fe(), quad, update_values | update_quadrature_points);
  const FEValuesExtractors::Vector ex(0);

  for (auto cell: g.get_dof_handler().active_cell_iterators()) {
    fe_values.reinit(cell);

    f.value_list(fe_values.get_quadrature_points(), f_values);
    fe_values[ex].get_function_values(g.get_coefficients(), g_values);

    for (unsigned int q = 0; q < n_q_points; ++q)
      Assert((f_values[q] - g_values[q]).norm() < 1.0e-6, ExcInternalError());
  }

  const double n = norm(g);
  const double exact_integral = std::sqrt(dim/3.0);
  Assert(abs(n - exact_integral) < dx, ExcInternalError());

  return true;
}


int main()
{
  const Point<2> p1(0.0, 0.0), p2(1.0, 1.0);
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);
  triangulation.refine_global(num_levels);

  const Discretization<2> discretization(triangulation, 1);

  Phi<2> phi;
  if (!test_field(discretization, phi)) return 1;

  F<2> f;
  if (!test_vector_field(discretization, f)) return 1;

  return 0;
}
