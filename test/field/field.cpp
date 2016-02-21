
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
bool test_field(
  const Triangulation<dim>& triangulation,
  const Function<dim>& phi
)
{
  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  Field<dim> psi = interpolate(triangulation, fe, dof_handler, phi);

  Quadrature<dim> quad(2);
  const unsigned int n_q_points = quad.size();

  const auto& coefficients = psi.get_coefficients();

  std::vector<double> phi_values(n_q_points), psi_values(n_q_points);
  FEValues<dim> fe_values(fe, quad, update_values | update_quadrature_points);
  const FEValuesExtractors::Scalar extractor(0);

  for (auto cell: dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);
    const std::vector<Point<dim> >& qs = fe_values.get_quadrature_points();
    phi.value_list(qs, phi_values);
    fe_values[extractor].get_function_values(coefficients, psi_values);

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
  const Triangulation<dim>& triangulation,
  const TensorFunction<1, dim>& f
)
{
  FESystem<dim> fe(FE_Q<dim>(1), dim);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  VectorField<dim> g = interpolate(triangulation, fe, dof_handler, f);

  Quadrature<dim> quad(2);
  const unsigned int n_q_points = quad.size();

  const auto& coefficients = g.get_coefficients();

  std::vector<Tensor<1, dim> > f_values(n_q_points), g_values(n_q_points);
  FEValues<dim> fe_values(fe, quad, update_values | update_quadrature_points);
  const FEValuesExtractors::Vector extractor(0);

  for (auto cell: dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);
    const std::vector<Point<dim> >& qs = fe_values.get_quadrature_points();
    f.value_list(qs, f_values);
    fe_values[extractor].get_function_values(coefficients, g_values);

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

  Phi<2> phi;
  if (!test_field(triangulation, phi)) return 1;

  F<2> f;
  if (!test_vector_field(triangulation, f)) return 1;

  return 0;
}
