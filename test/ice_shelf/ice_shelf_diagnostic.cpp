
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


void test_derivatives(
  const IceShelf& ice_shelf,
  const Field<2>& h,
  const Field<2>& theta,
  const VectorField<2>& u,
  const VectorField<2>& du,
  const bool verbose
)
{
  const double P = ice_shelf.action(h, theta, u);
  const DualVectorField<2> dP = ice_shelf.derivative(h, theta, u);
  const SparseMatrix<double> d2P = ice_shelf.hessian(h, theta, u);

  const double linear_term = inner_product(dP, du);

  const size_t num_trials = 12;
  std::vector<double> errors(num_trials);
  for (size_t k = 0; k < num_trials; ++k) {
    const double eps = 1.0 / pow(2.0, k);

    const VectorField<2> v = u + eps * du;
    const double action_exact = ice_shelf.action(h, theta, v);

    const double action_approx = P + eps * linear_term;
    const double error = std::abs(action_exact - action_approx);

    errors[k] = error/(eps * std::abs(P));
  }

  if (verbose) {
    std::cout << "Error in local quadratic approximation of shallow shelf "
              << "action functional: " << std::endl;
    for (size_t k = 0; k < num_trials; ++k)
      std::cout << errors[k] << std::endl;

    std::cout << std::endl;
  }

  check(icepack::testing::is_decreasing(errors));
}


void test_hessians(
  const IceShelf& ice_shelf,
  const Field<2>& h,
  const Field<2>& theta,
  const VectorField<2>& u,
  const VectorField<2>& du,
  const bool verbose
)
{
  const DualVectorField<2> dP = ice_shelf.derivative(h, theta, u);
  const SparseMatrix<double> d2P = ice_shelf.hessian(h, theta, u);

  DualVectorField<2> d2P_times_du(dP.get_discretization());
  d2P.vmult(d2P_times_du.get_coefficients(), du.get_coefficients());

  const size_t num_trials = 12;
  std::vector<double> errors(num_trials);
  for (size_t k = 0; k < num_trials; ++k) {
    const double eps = 1.0 / pow(2.0, k);
    const VectorField<2> v = u + eps * du;

    const DualVectorField<2> dP_exact = ice_shelf.derivative(h, theta, v);
    const DualVectorField<2> dP_approx = dP + eps * d2P_times_du;

    const DualVectorField<2> ddP = dP_exact - dP_approx;
    errors[k] = norm(ddP);
  }

  if (verbose) {
    std::cout << "Error in linearization of shallow shelf equations: "
              << std::endl;
    for (size_t k = 0; k < num_trials; ++k)
      std::cout << errors[k] << std::endl;

    std::cout << std::endl;
  }

  check(icepack::testing::is_decreasing(errors));
}



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
   * Test computing the power dissipation rate of the candidate fields
   */

  const double area = width * length;
  const double B = ice_shelf.constitutive_tensor.rheology.B(temp);
  const double n = ice_shelf.constitutive_tensor.rheology.n;
  const double h_integral =
    (std::pow(h0, n + 3) - std::pow(h0 - delta_h, n + 3))/((n + 3) * delta_h);
  const double power =
    area * std::pow(rho * gravity, n + 1) / std::pow(4 * B, n) * h_integral;
  const double ice_stress_P = n/(n + 1) * power / 2;
  const double driving_stress_P = -0.5 * power;
  const double P_exact = ice_stress_P + driving_stress_P;

  const double P = ice_shelf.action(h, theta, u_true);

  if (verbose)
    std::cout << "Energy dissipation per unit area: " << P / area << std::endl;

  check_real(P, P_exact, std::pow(dx, p + 1) * std::abs(P_exact));


  /**
   * Test computing the residual of a candidate velocity
   */

  const DualVectorField<2> tau = ice_shelf.driving_stress(h);
  const DualVectorField<2> dP = ice_shelf.derivative(h, theta, u_true);

  // Residual of the exact solution should be < dx^2.
  check_real(norm(dP)/norm(tau), 0, dx*dx);


  /**
   * Check that the action can be approximated locally to 2nd order using the
   * derivative and Hessian.
   */

  test_derivatives(ice_shelf, h, theta, u_true, u0 - u_true, verbose);
  test_derivatives(ice_shelf, h, theta, u0, u_true - u0, verbose);

  test_hessians(ice_shelf, h, theta, u_true, u0 - u_true, verbose);
  test_hessians(ice_shelf, h, theta, u0, u_true - u0, verbose);

  /**
   * Test the diagnostic solve procedure
   */

  const VectorField<2> u = ice_shelf.diagnostic_solve(h, theta, u0);
  check_fields(u, u_true, std::pow(dx, p+1));


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
