
#include <deal.II/grid/grid_generator.h>

#include <icepack/physics/constants.hpp>
#include <icepack/physics/viscosity.hpp>
#include <icepack/glacier_models/ice_stream.hpp>

using namespace dealii;
using namespace icepack;


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


class Beta : public Function<2>
{
public:
  Beta() {}

  double value(const Point<2>&, const unsigned int = 0) const
  {
    return 0.5;
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

  const Field<2> beta = ssa.interpolate(Beta());


  const double u0 = 100.0;
  const double temp = 263.15;
  const double A = pow(rho * gravity * h0 / 4, 3) * rate_factor(temp);
  const double tau = delta_h / (h0 * A);
  const double gamma = delta_h / h0;
  const VectorField<2> v = ssa.interpolate(Velocity(u0, tau, gamma, length));

  const VectorField<2> u = ssa.diagnostic_solve(s, h, beta, v);

  if (verbose) {
    u.write("u.ucd", "u");
  }

  return 0;
}
