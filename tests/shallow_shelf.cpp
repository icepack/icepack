
#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include "icepack/glacier_models/shallow_shelf.hpp"

using namespace dealii;
using namespace icepack;

/**
 * This program tests the correctness of the our implementation of the shallow
 * shelf approximation using the method of manufactured solutions.
 * Constant ice flow and constant thickness are a simple solution of SSA.
 */


const double eps = 0.2;
const double temp = 263.15;
const double surf = 8 * viscosity(temp, eps) * eps / (rho_ice * gravity);


class BoundaryVelocity : public TensorFunction<1, 2>
{
public:
  BoundaryVelocity () : TensorFunction<1, 2>() {}
  virtual Tensor<1, 2> value (const Point<2>& x) const
  {
    Tensor<1, 2> v;
    v[0] = 100.0 + eps * x[0];
    v[1] = 0.0;
    return v;
  }
};


int main()
{
  dealii::deallog.depth_console (0);

  const Point<2> p1(0.0, 0.0), p2(2000.0, 500.0);
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);

  // Mark the right side of the rectangle as the ice front
  for (auto cell: triangulation.active_cell_iterators()) {
    for (unsigned int face_number = 0;
         face_number < GeometryInfo<2>::faces_per_cell;
         ++face_number)
      if (cell->face(face_number)->center()(0) > 1999.0)
        cell->face(face_number)->set_boundary_id (1);
  }

  triangulation.refine_global(2);

  auto surface = ConstantFunction<2>(surf);
  auto bed     = ConstantFunction<2>(-2000.0);
  BoundaryVelocity boundary_velocity;

  ShallowShelf shallow_shelf(triangulation, surface, bed, boundary_velocity);
  shallow_shelf.setup_system(true);

  shallow_shelf.run();

  Vector<double> difference(triangulation.n_cells());

  VectorTools::integrate_difference
    (shallow_shelf.dof_handler,
     shallow_shelf.solution,
     VectorFunctionFromTensorFunction<2> (boundary_velocity),
     difference,
     shallow_shelf.quadrature_formula,
     VectorTools::Linfty_norm);

  const double error = difference.linfty_norm();
  if (error > 1.0e-10) {
    std::cout << error << std::endl;
    return 1;
  }

  return 0;
}
