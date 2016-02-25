
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

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


/**
 * an accumulation field for which the linear ice ramp is a steady state
 */
class Accumulation : public Function<2>
{
public:
  Accumulation()
    :
    h(Thickness()),
    u(Velocity())
  {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    const double q = 1 - delta_h / h0 * x[0] / length;
    const double du_dx = A * pow(q, 3);
    const double dh_dx = -delta_h / length;

    return h.value(x) * du_dx + u.value(x)[0] * dh_dx;
  }

  const Thickness h;
  const Velocity u;
};


int main(int argc, char ** argv)
{
  const bool verbose = argc == 2 &&
    (strcmp(argv[1], "-v") == 0 ||
     strcmp(argv[1], "--verbose") == 0);

  /**
   * Create a triangulation on which to solve PDEs
   */

  Triangulation<2> tria;
  const Point<2> p1(0.0, 0.0), p2(length, width);
  GridGenerator::hyper_rectangle(tria, p1, p2);

  // Mark the right side of the rectangle as the ice front
  for (auto cell: tria.active_cell_iterators()) {
    for (unsigned int face_number = 0;
         face_number < GeometryInfo<2>::faces_per_cell;
         ++face_number)
      if (cell->face(face_number)->center()(0) > length - 1.0)
        cell->face(face_number)->set_boundary_id(1);
  }

  const unsigned int num_levels = 5;
  tria.refine_global(num_levels);
  const double dx = 1.0 / (1 << num_levels);


  /**
   * Create a model object and input data
   */

  IceShelf ice_shelf(tria, 1);

  Field<2> h0 = ice_shelf.interpolate(Thickness());
  Field<2> theta = ice_shelf.interpolate(Temperature());
  Field<2> a = ice_shelf.interpolate(Accumulation());
  VectorField<2> u = ice_shelf.interpolate(Velocity());

  const double mesh_size = dealii::GridTools::minimal_cell_diameter(tria);
  const Point<2> x(width/2, length - 0.25);
  const double dt = mesh_size / Velocity().value(x)[0] / 2;

  Field<2> h = ice_shelf.prognostic_solve(dt, h0, a, u);

  if (verbose) {
    h0.write("h0.ucd", "h0");
    h.write("h.ucd", "h");
    u.write("u.ucd", "u");
    a.write("a.ucd", "a");
  }

  Assert(dist(h, h0) / norm(h0) < dx, ExcInternalError());

  return 0;
}
