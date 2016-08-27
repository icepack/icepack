
#include <fstream>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <icepack/physics/constants.hpp>
#include <icepack/physics/viscosity.hpp>
#include <icepack/glacier_models/ice_shelf.hpp>
#include "../testing.hpp"

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
  /**
   * Parse command-line arguments
   */
  std::set<std::string> args = testing::get_cmdline_args(argc, argv);

  const bool verbose = args.count("-v") || args.count("--verbose");
  const bool refined = args.count("--refined");
  const int q2 = args.count("--q2");


  /**
   * Create a triangulation on which to solve PDEs
   */
  const unsigned int levels = 5 - q2;
  Triangulation<2> tria = testing::rectangular_glacier(length, width, levels);

  // Dimensionless mesh resolution; the finite element solution is accurate to
  // order O(dx^{p+1}), where dx is the mesh resolution and p is the polynomial
  // order, i.e. 1 if we're using piecewise bilinear elements and 2 if we're
  // using piecewise biquadratic.
  const double dx = 1.0 / (1 << levels);

  // If this test is using a non-uniform grid, refine everything on the right
  // side of the domain.
  if (refined) {
    Vector<double> refinement_criteria(tria.n_active_cells());
    for (const auto cell: tria.cell_iterators()) {
      const unsigned int index = cell->index();
      Point<2> x = cell->barycenter();
      refinement_criteria[index] = x[0] / length;
    }

    GridRefinement::refine(tria, refinement_criteria, 0.5);
    tria.execute_coarsening_and_refinement();

    if (verbose) {
      GridOut grid_out;
      std::ofstream out("grid.msh");
      grid_out.write_msh(tria, out);
    }
  }


  /**
   * Create a model object and input data
   */

  // The polynomial order is 1 by default, 2 if we use biquadratic elements
  const unsigned int p = 1 + q2;
  IceShelf ice_shelf(tria, p);

  Field<2> h = ice_shelf.interpolate(Thickness());
  Field<2> theta = ice_shelf.interpolate(Temperature());
  VectorField<2> u_true = ice_shelf.interpolate(Velocity());
  VectorField<2> u0 = ice_shelf.interpolate(BoundaryVelocity());


  /**
   * Test computing the residual of a candidate velocity
   */

  const DualVectorField<2> tau = ice_shelf.driving_stress(h);
  const DualVectorField<2> r = ice_shelf.residual(h, theta, u_true, tau);

  // Residual of the exact solution should be < dx^2.
  check(norm(r)/norm(tau) < dx*dx);


  /**
   * Test the diagnostic solve procedure
   */

  const VectorField<2> u = ice_shelf.diagnostic_solve(h, theta, u0);
  check(dist(u, u_true)/norm(u_true) < std::pow(dx, p+1));


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
