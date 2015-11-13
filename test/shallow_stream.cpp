
#include <deal.II/grid/grid_generator.h>

#include <icepack/glacier_models/shallow_stream.hpp>

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


class Surface : public Function<2>
{
public:
  Surface() {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    return (1 - rho_ice/rho_water) * (h0 - delta_h/length * x[0]);
  }
};

class Thickness : public Function<2>
{
public:
  Thickness() {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    return h0 - delta_h/length * x[0];
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


int main()
{
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

  triangulation.refine_global(3);

  ShallowStream ssa(triangulation, 1);
  const Surface _s;
  const Thickness _h;

  Field<2> s = ssa.interpolate(_s);
  Field<2> h = ssa.interpolate(_h);
  Field<2> beta = ssa.interpolate(ZeroFunction<2>());
  VectorField<2> u0 = ssa.interpolate(Velocity());

  VectorField<2> tau = ssa.driving_stress(s, h);

  VectorField<2> u = ssa.diagnostic_solve(s, h, beta, u0);

  return 0;
}
