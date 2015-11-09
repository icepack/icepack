
#include <deal.II/grid/grid_generator.h>

#include <icepack/glacier_models/shallow_stream.hpp>

using namespace dealii;
using namespace icepack;

const double length = 2000.0;
const double width = 500.0;

int main()
{
  Triangulation<2> triangulation;
  const Point<2> p1(0.0, 0.0), p2(length, width);
  GridGenerator::hyper_rectangle(triangulation, p1, p2);
  triangulation.refine_global(3);

  ShallowStream ssa(triangulation, 1);
  const auto& vector_pde = ssa.get_vector_pde_skeleton();

  const FiniteElement<2>& fe = vector_pde.get_fe();
  const DoFHandler<2>& dof_handler = vector_pde.get_dof_handler();
  const QGauss<2> quadrature(2);

  FEValues<2> fe_values(fe, quadrature, update_quadrature_points);

  for (auto cell: dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);

    const auto& xs = fe_values.get_quadrature_points();

    for (auto x: xs) std::cout << x << "  ";
    std::cout << std::endl;
  }

  return 0;
}
