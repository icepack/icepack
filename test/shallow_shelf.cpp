
#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <icepack/glacier_models/shallow_shelf.hpp>

using namespace dealii;
using namespace icepack;

/**
 * This program tests the correctness of the our implementation of the shallow
 * shelf approximation using the method of manufactured solutions.
 * We use the exact solution for the ice ramp from ch. 6.4 in Greve & Blatter.
 */


const double rho = rho_ice * (1 - rho_ice / rho_water);
const double temp = 263.15;
const double A = pow(rho * gravity / 4, 3) * rate_factor(temp);


const double u0 = 100;
const double h0 = 500;
const double delta_h = 100.0;
const double length = 2000.0;
const double width = 500.0;


class BoundaryVelocity : public TensorFunction<1, 2>
{
public:
  BoundaryVelocity () : TensorFunction<1, 2>() {}

  Tensor<1, 2> value (const Point<2>& x) const
  {
    const double q = 1 - pow(1 - delta_h * x[0] / (length * h0), 4);

    // This is the exact solution to the PDE.
    Tensor<1, 2> v;
    v[0] = u0 + 0.25 * A * q * length * pow(h0, 4) / delta_h;
    v[1] = 0.0;

    // Fudge factor so the initial guess isn't the exact solution.
    const double px = x[0] / length, py = x[1] / width;
    const double p = px * (1 - px) * py * (1 - py);
    v[0] += p * 10.0;
    v[1] += p * (0.5 - py) * 5.0;

    return v;
  }
};


class SurfaceElevation : public Function<2>
{
public:
  SurfaceElevation() : Function<2>() {}

  double value (const Point<2>& x, const unsigned int) const
  {
    return (1 - rho_ice / rho_water) * (h0 - delta_h / length * x[0]);
  }

  Tensor<1, 2> gradient(const Point<2>&, const unsigned int) const
  {
    Tensor<1, 2> ds;
    ds[0] = -(1 - rho_ice / rho_water) * delta_h / length;
    ds[1] = 0.0;

    return ds;
  }
};


int main()
{
  dealii::deallog.depth_console (0);

  const Point<2> p1(0.0, 0.0), p2(length, width);
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);

  // Mark the right side of the rectangle as the ice front
  for (auto cell: triangulation.active_cell_iterators()) {
    for (unsigned int face_number = 0;
         face_number < GeometryInfo<2>::faces_per_cell;
         ++face_number)
      if (cell->face(face_number)->center()(0) > length - 1.0)
        cell->face(face_number)->set_boundary_id (1);
  }

  triangulation.refine_global(2);

  auto bed         = ConstantFunction<2>(-2000.0);
  auto temperature = ConstantFunction<2>(temp);
  auto friction    = ZeroFunction<2>();
  SurfaceElevation surface;
  BoundaryVelocity boundary_velocity;

  ShallowShelf shallow_shelf(triangulation, surface, bed, temperature,
                             friction, boundary_velocity);

  shallow_shelf.diagnostic_solve(1.0e-8);

  Vector<double>& solution = shallow_shelf.solution;
  Vector<double> difference(triangulation.n_cells());

  VectorTools::integrate_difference
    (shallow_shelf.dof_handler,
     solution,
     VectorFunctionFromTensorFunction<2> (boundary_velocity),
     difference,
     shallow_shelf.quadrature_formula,
     VectorTools::Linfty_norm);

  const double error = difference.linfty_norm() / solution.linfty_norm();
  if (error > 1.0e-2) {
    std::cout << error << std::endl;
    return 1;
  }

  return 0;
}
