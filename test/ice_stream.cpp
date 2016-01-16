
#include <deal.II/grid/grid_generator.h>

#include <icepack/physics/constants.hpp>
#include <icepack/physics/viscosity.hpp>
#include <icepack/glacier_models/ice_stream.hpp>

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


class Velocity : public TensorFunction<1, 2>
{
public:
  Velocity(const double u0, const double tau, const double gamma, const double length)
    :
    u0(u0),
    tau(tau),
    gamma(gamma),
    length(length)
  {}

  Tensor<1, 2> value(const Point<2>& x) const
  {
    Tensor<1, 2> v;
    const double q = pow(1 - gamma * x[0] / length, 4);
    v[0] = u0 + length / (4 * tau) * (1 - q);
    v[1] = 0.0;

    return v;
  }

  const double u0, tau, gamma, length;
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
    const double B = std::pow(rate_factor(temp), -1.0/3);
    const double q = v.gamma * h.h0 + h.delta_h * (1 - v.gamma * x[0] / v.length);
    return -2 * B / v.length * std::pow(v.gamma / v.tau, 1.0/3) * q;
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
  Beta(const double m, const Velocity& v, const Thickness& h, const Surface& s)
    :
    m(m),
    v(v),
    h(h),
    s(s),
    M(v, h),
    tau_d(h, s)
  {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    return (M.value(x) + tau_d.value(x)) / std::pow(v.value(x)[0], 1.0/m);
  }

  const double m;
  const Velocity& v;
  const Thickness& h;
  const Surface& s;
  const MembraneStressDivergence M;
  const DrivingStress tau_d;
};


int main(int argc, char ** argv)
{
  const bool verbose = argc == 2 &&
    (strcmp(argv[1], "-v") == 0 ||
     strcmp(argv[1], "--verbose") == 0);

  /**
   * Create a triangulation on which to solve PDEs
   */

  const double length = 2000, width = 500;
  Triangulation<2> triangulation;
  const Point<2> p1(0.0, 0.0), p2(length, width);
  GridGenerator::hyper_rectangle(triangulation, p1, p2);

  for (auto cell: triangulation.active_cell_iterators()) {
    for (unsigned int face_number = 0;
         face_number < GeometryInfo<2>::faces_per_cell;
         ++face_number)
      if (cell->face(face_number)->center()(0) > length - 1.0)
        cell->face(face_number)->set_boundary_id(1);
  }

  const unsigned int num_levels = 5;
  triangulation.refine_global(num_levels);


  /**
   * Create a model object and input data
   */

  IceStream ssa(triangulation, 1);

  const double h0 = 500, delta_h = 100;
  const Thickness thickness(h0, delta_h, length);
  const Field<2> h = ssa.interpolate(thickness);

  const double height_above_flotation = 50.0;
  const double rho = rho_ice * (1 - rho_ice / rho_water),
    s0 = (1 - rho_ice/rho_water) * h0 + height_above_flotation,
    delta_s = (1 - rho_ice/rho_water) * delta_h + height_above_flotation;
  const Surface surface(s0, delta_s, length);
  const Field<2> s = ssa.interpolate(surface);

  const double u0 = 100.0;
  const double A = pow(rho * gravity * h0 / 4, 3) * rate_factor(temp);
  const double tau = delta_h / (h0 * A);
  const double gamma = delta_h / h0;
  const Velocity velocity(u0, tau, gamma, length);
  const VectorField<2> v = ssa.interpolate(velocity);

  const Beta _beta(3.0, velocity, thickness, surface);
  const Field<2> beta = ssa.interpolate(_beta);

  const VectorField<2> u = ssa.diagnostic_solve(s, h, beta, v);

  if (verbose) {
    u.write("u.ucd", "u");
  }

  return 0;
}
