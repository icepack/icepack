
#include <deal.II/grid/grid_generator.h>

#include <icepack/physics/constants.hpp>
#include <icepack/physics/viscosity.hpp>
#include <icepack/glacier_models/ice_shelf.hpp>

using namespace dealii;
using namespace icepack;

const double rho = rho_ice * (1 - rho_ice / rho_water);
const double temp = 263.15;
const double A = pow(rho * gravity / 4, 3) * rate_factor(temp);

const double u0 = 100;
const double length = 2000;
const double width = 500;
const double h0 = 500;
const double delta_h = 100;


class Thickness : public Function<2>
{
public:
  Thickness() {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    return h0 - delta_h/length * x[0];
  }
};

class Temperature : public Function<2>
{
public:
  Temperature() {}

  double value(const Point<2>&, const unsigned int = 0) const
  {
    return temp;
  }
};

class Velocity : public TensorFunction<1, 2>
{
public:
  Velocity() {}

  Tensor<1, 2> value(const Point<2>& x) const
  {
    const double q = 1 - pow(1 - delta_h * x[0] / (length * h0), 4);

    Tensor<1, 2> v;
    v[0] = u0 + 0.25 * A * q * length * pow(h0, 4) / delta_h;
    v[1] = 0.0;

    return v;
  }
};

class BoundaryVelocity : public TensorFunction<1, 2>
{
public:
  BoundaryVelocity() {}

  Tensor<1, 2> value(const Point<2>& x) const
  {
    const double q = 1 - pow(1 - delta_h * x[0] / (length * h0), 4);

    Tensor<1, 2> v;
    v[0] = u0 + 0.25 * A * q * length * pow(h0, 4) / delta_h;
    v[1] = 0.0;

    // Fudge factor so this isn't the same as the exact solution
    const double px = x[0] / length;
    const double ax = px * (1 - px);
    const double py = x[1] / width;
    const double ay = py * (1 - py);
    v[1] += ax * ay * (0.5 - py) * 500.0;
    v[0] += ax * ay * 500.0;

    return v;
  }
};



int main(int argc, char ** argv)
{
  const bool verbose = argc == 2 &&
    (strcmp(argv[1], "-v") == 0 ||
     strcmp(argv[1], "--verbose") == 0);

  /**
   * Create a triangulation on which to solve PDEs
   */

  Triangulation<2> triangulation;
  const Point<2> p1(0.0, 0.0), p2(length, width);
  GridGenerator::hyper_rectangle(triangulation, p1, p2);

  // Mark the right side of the rectangle as the ice front
  for (auto cell: triangulation.active_cell_iterators()) {
    for (unsigned int face_number = 0;
         face_number < GeometryInfo<2>::faces_per_cell;
         ++face_number)
      if (cell->face(face_number)->center()(0) > length - 1.0)
        cell->face(face_number)->set_boundary_id(1);
  }

  const unsigned int num_levels = 5;
  triangulation.refine_global(num_levels);

  // Dimensionless mesh resolution; finite element solution is only
  // O(dx^2) accurate.
  const double dx = 1.0 / (1 << num_levels);


  /**
   * Create a model object and input data
   */

  IceShelf ice_shelf(triangulation, 1);

  Field<2> h = ice_shelf.interpolate(Thickness());
  Field<2> theta = ice_shelf.interpolate(Temperature());
  VectorField<2> u_true = ice_shelf.interpolate(Velocity());
  VectorField<2> u0 = ice_shelf.interpolate(BoundaryVelocity());


  /**
   * Test computing the residual of a candidate velocity
   */

  const VectorField<2> tau = ice_shelf.driving_stress(h);
  const VectorField<2> r = ice_shelf.residual(h, theta, u_true, tau);
  const Vector<double>& Tau = tau.get_coefficients();
  const Vector<double>& R = r.get_coefficients();

  // Residual of the exact solution should be < dx^2.
  Assert(R.l2_norm() / Tau.l2_norm() < dx*dx, ExcInternalError());


  /**
   * Test the diagnostic solve procedure
   */

  const VectorField<2> u = ice_shelf.diagnostic_solve(h, theta, u0);
  Assert(dist(u, u_true)/norm(u_true) < dx*dx, ExcInternalError());


  /**
   * Write out the solution to a file if running in verbose mode
   */

  if (verbose) {
    std::cout << "Relative initial error: "
              << dist(u0, u_true)/norm(u_true) << std::endl;

    std::cout << "Relative final error:   "
              << dist(u, u_true)/norm(u_true) << std::endl;

    u0.write("u0.ucd", "u0");
    u_true.write("u_true.ucd", "u_true");
    u.write("u.ucd", "u");
  }

  return 0;
}
