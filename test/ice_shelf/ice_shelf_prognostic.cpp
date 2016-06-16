
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
} thickness;

class DhDx : public Function<2>
{
public:
  DhDx() {}

  double value(const Point<2>&, const unsigned int = 0) const
  {
    return -delta_h/length;
  }
} dh_dx;

class Temperature : public Function<2>
{
public:
  Temperature() {}

  double value(const Point<2>&, const unsigned int = 0) const
  {
    return temp;
  }
} temperature;

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
} velocity;

class DuDx : public Function<2>
{
public:
  DuDx() {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    const double q = 1 - delta_h / h0 * x[0] / length;
    return A * pow(h0 * q, 3);
  }
} du_dx;

/**
 * an accumulation field for which the linear ice ramp is a steady state
 */
class Accumulation : public Function<2>
{
public:
  Accumulation() {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    return thickness.value(x) * du_dx.value(x)
      + velocity.value(x)[0] * dh_dx.value(x);
  }
} accumulation;


int main(int argc, char ** argv)
{
  const bool verbose = argc == 2 &&
    (strcmp(argv[1], "-v") == 0 ||
     strcmp(argv[1], "--verbose") == 0);


  /**
   * Create a model object and input data
   */
  Triangulation<2> tria = testing::rectangular_glacier(length, width);
  const double dx = dealii::GridTools::minimal_cell_diameter(tria);

  IceShelf ice_shelf(tria, 1);

  Field<2> h0 = ice_shelf.interpolate(thickness);
  Field<2> theta = ice_shelf.interpolate(temperature);
  Field<2> a = ice_shelf.interpolate(accumulation);
  VectorField<2> u = ice_shelf.interpolate(velocity);

  const Point<2> x(width/2, length - 0.25);
  const double dt = dx / Velocity().value(x)[0] / 2;

  Field<2> h(h0);

  for (size_t k = 0; k < 32; ++k)
    h = ice_shelf.prognostic_solve(dt, h, a, u);

  Field<2> dh_dt = ice_shelf.dh_dt(h0, a, u);

  if (verbose) {
    h0.write("h0.ucd", "h0");
    h.write("h.ucd", "h");
    u.write("u.ucd", "u");
    a.write("a.ucd", "a");
    dh_dt.write("dh_dt.ucd", "dh_dt");
  }

  Assert(dist(h, h0) / norm(h0) < dx, ExcInternalError());

  return 0;
}
