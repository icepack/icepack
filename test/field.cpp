
#include <iostream>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_values.h>

#include <icepack/field.hpp>


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
bool test_field(
  const Triangulation<dim>& triangulation,
  const Function<dim>& phi
)
{
  FE_Q<dim> finite_element(1);
  Field<dim> psi = interpolate(triangulation, finite_element, phi);

  Quadrature<dim> quad(2);
  const unsigned int n_q_points = quad.size();

  const auto& dof_handler = psi.get_dof_handler();
  const auto& coefficients = psi.get_coefficients();

  std::vector<double> phi_values(n_q_points);
  std::vector<double> psi_values(n_q_points);
  FEValues<dim> fe_values(finite_element, quad,
                          update_values | update_quadrature_points);
  const FEValuesExtractors::Scalar extractor(0);

  for (auto cell: dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);
    const std::vector<Point<dim> >& qs = fe_values.get_quadrature_points();
    phi.value_list(qs, phi_values);
    fe_values[extractor].get_function_values(coefficients, psi_values);

    for (unsigned int q = 0; q < n_q_points; ++q)
      Assert(abs(phi_values[q] - psi_values[q]) < 1.0e-6, ExcInternalError());
  }

  return true;
}


int main()
{
  const Point<2> p1(0.0, 0.0), p2(1.0, 1.0);
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);
  triangulation.refine_global(3);

  Phi<2> phi;

  if (!test_field(triangulation, phi)) return 1;

  return 0;
}
