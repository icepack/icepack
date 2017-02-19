
#include <icepack/physics/constants.hpp>
#include <icepack/physics/viscosity.hpp>
#include <icepack/glacier_models/ice_shelf.hpp>
#include "../testing.hpp"

using namespace dealii;
using namespace icepack;

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

class Velocity : public TensorFunction<1, 2>
{
public:
  Velocity() {}

  Tensor<1, 2> value(const Point<2>&) const
  {

    Tensor<1, 2> v;
    v[0] = u0;
    v[1] = 0.0;

    return v;
  }
} velocity;

class DuDx : public Function<2>
{
public:
  DuDx() {}

  double value(const Point<2>&, const unsigned int = 0) const
  {
    return 0.0;
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

  // Create a model object and input data
  const size_t num_levels = 5;
  Triangulation<2> tria = testing::rectangular_glacier(length, width, num_levels);
  IceShelf ice_shelf(tria, 1);

  Field<2> h0 = ice_shelf.interpolate(thickness);
  Field<2> a = ice_shelf.interpolate(accumulation);
  VectorField<2> u = ice_shelf.interpolate(velocity);

  // Pick a timestep which will satisfy the Courant-Friedrichs-Lewy condition
  const Point<2> x(width/2, length - 0.25);
  const double max_speed = velocity.value(x)[0];
  const double dt =
      dealii::GridTools::minimal_cell_diameter(tria) / max_speed / 2;
  const double residence_time = length / max_speed;

  Field<2> h(h0);

  // Pick the number of timesteps so that ice from the inflow boundary will
  // propagate most of the way through the domain
  size_t num_timesteps = (size_t)(residence_time / dt);
  if (verbose)
    std::cout << "Number of timesteps: " << num_timesteps << std::endl;

  // Propagate the thickness field a few timesteps forward
  for (size_t k = 0; k < num_timesteps; ++k)
    h = ice_shelf.prognostic_solve(dt, h, a, u);

  Field<2> dh_dt = ice_shelf.dh_dt(h0, a, u);

  if (verbose) {
    h0.write("h0.ucd", "h0");
    h.write("h.ucd", "h");
    u.write("u.ucd", "u");
    a.write("a.ucd", "a");
    dh_dt.write("dh_dt.ucd", "dh_dt");
  }

  // The accumulation rate was chosen so that, with the given ice thickness and
  // velocity, the flow would be a steady state; check that the final thickness
  // is reasonably close to the initial thickness.
  const double dx = 1.0 / (1 << num_levels);
  check_fields(h, h0, std::max(dx * dx, dt / residence_time));

  return 0;
}
