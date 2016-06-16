
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>

#include <icepack/physics/constants.hpp>
#include <icepack/physics/viscosity.hpp>
#include <icepack/glacier_models/ice_stream.hpp>

#include "../testing.hpp"

using std::pow;
using namespace dealii;
using namespace icepack;


const double temp = 263.15;

class Surface : public Function<2>
{
public:
  Surface(const double s0, const double delta_s, const double length)
    :
    s0(s0),
    delta_s(delta_s),
    length(length)
  {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    return s0 - delta_s * x[0] / length;
  }

  const double s0, delta_s, length;
};


class Thickness : public Function<2>
{
public:
  Thickness(const double h0, const double delta_h, const double length)
    :
    h0(h0),
    delta_h(delta_h),
    length(length)
  {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    return h0 - x[0] / length * delta_h;
  }

  const double h0, delta_h, length;
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
  Velocity(const double u0, const double alpha, const double gamma, const double length)
    :
    u0(u0),
    alpha(alpha),
    gamma(gamma),
    length(length)
  {}

  Tensor<1, 2> value(const Point<2>& x) const
  {
    Tensor<1, 2> v;
    const double q = pow(1 - gamma * x[0] / length, 4);
    v[0] = u0 + length / (4 * alpha) * (1 - q);
    v[1] = 0.0;

    return v;
  }

  const double u0, alpha, gamma, length;
};


class InitialVelocity : public TensorFunction<1, 2>
{
public:
  InitialVelocity(
    const double u0,
    const double alpha,
    const double gamma,
    const double length,
    const double width
  )
    :
    u0(u0),
    alpha(alpha),
    gamma(gamma),
    length(length),
    width(width)
  {}

  Tensor<1, 2> value(const Point<2>& x) const
  {
    Tensor<1, 2> v;
    const double q = pow(1 - gamma * x[0] / length, 4);
    v[0] = u0 + length / (4 * alpha) * (1 - q);
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

  const double u0, alpha, gamma, length, width;
};


/**
 * This function is the x-x component of the membrane stress divergence for the
 * given ice velocity and thickness. For the math, see my thesis.
 */
class MembraneStressDivergence : public Function<2>
{
public:
  MembraneStressDivergence(const Velocity& v, const Thickness& h)
    :
    v(v),
    h(h)
  {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    const double B = pow(rate_factor(temp), -1.0/3);
    const double dh_dx = -h.delta_h / h.length;
    const double q = (1 - v.gamma * x[0] / v.length) * dh_dx - v.gamma / v.length * h.value(x);
    return 2 * B * pow(v.gamma / v.alpha, 1.0/3) * q;
  }

  const Velocity& v;
  const Thickness& h;
};


class DrivingStress : public Function<2>
{
public:
  DrivingStress(const Thickness& h, const Surface& s)
    :
    h(h),
    s(s)
  {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    const double grad_s = -s.delta_s / s.length;
    return -rho_ice * gravity * h.value(x) * grad_s;
  }

  const Thickness& h;
  const Surface& s;
};


class Beta : public Function<2>
{
public:
  Beta(
    const double m,
    const double u0,
    const double tau0,
    const Velocity& v,
    const Thickness& h,
    const Surface& s
  )
    :
    m(m),
    u0(u0),
    tau0(tau0),
    v(v),
    h(h),
    s(s),
    M(v, h),
    tau_d(h, s)
  {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    const double C = (M.value(x) + tau_d.value(x)) / pow(v.value(x)[0], 1.0/m);
    const double beta = std::log(C / tau0 * pow(u0, 1.0/m));
    return beta;
  }

  const double m, u0, tau0;
  const Velocity& v;
  const Thickness& h;
  const Surface& s;
  const MembraneStressDivergence M;
  const DrivingStress tau_d;
};


int main(int argc, char ** argv)
{
  std::set<std::string> args = testing::get_cmdline_args(argc, argv);

  const bool verbose = args.count("-v") || args.count("--verbose");
  const bool refined = args.count("--refined");
  const int q2 = args.count("--q2");

  /**
   * Create a triangulation on which to solve PDEs
   */

  const double length = 2000, width = 500;
  const unsigned int levels = 5 - q2;
  Triangulation<2> tria = testing::rectangular_glacier(length, width, levels);
  const double dx = 1.0 / (1 << levels);

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
  IceStream ice_stream(tria, p);

  const double h0 = 500, delta_h = 100;
  const Thickness thickness(h0, delta_h, length);
  const Field<2> h = ice_stream.interpolate(thickness);

  const double height_above_flotation = 50.0;
  const double rho = rho_ice * (1 - rho_ice / rho_water),
    s0 = (1 - rho_ice/rho_water) * h0 + height_above_flotation,
    delta_s = (1 - rho_ice/rho_water) * delta_h + height_above_flotation;
  const Surface surface(s0, delta_s, length);
  const Field<2> s = ice_stream.interpolate(surface);

  const Field<2> theta = ice_stream.interpolate(Temperature());

  const double u0 = 100.0;
  const double A = pow(rho * gravity * h0 / 4, 3) * rate_factor(temp);
  const double alpha = delta_h / (h0 * A);
  const double gamma = delta_h / h0;

  const Velocity velocity(u0, alpha, gamma, length);
  const VectorField<2> u_true = ice_stream.interpolate(velocity);

  const VectorField<2> u_init =
    ice_stream.interpolate(InitialVelocity(u0, alpha, gamma, length, width));

  const Beta _beta(
    ice_stream.basal_shear.m,
    ice_stream.basal_shear.u0,
    ice_stream.basal_shear.tau0,
    velocity,
    thickness,
    surface
  );
  const Field<2> beta = ice_stream.interpolate(_beta);

  /**
   * Test computing the model residual
   */

  const DualVectorField<2> tau = ice_stream.driving_stress(s, h);
  const DualVectorField<2> r = ice_stream.residual(s, h, theta, beta, u_true, tau);

  // Residual of the exact solution should be < dx^2.
  Assert(norm(r)/norm(tau) < dx*dx, ExcInternalError());


  /**
   * Test the diagnostic solve procedure
   */

  const VectorField<2> u =
    ice_stream.diagnostic_solve(s, h, theta, beta, u_init);
  Assert(dist(u, u_true)/norm(u_true) < dx*dx, ExcInternalError());


  /**
   * Write out the solution to a file if running in verbose mode
   */

  if (verbose) {
    u.write("u.ucd", "u");
    u_true.write("u0.ucd", "u0");
  }

  return 0;
}
