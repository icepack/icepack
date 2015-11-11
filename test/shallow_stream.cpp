
#include <deal.II/grid/grid_generator.h>

#include <icepack/glacier_models/shallow_stream.hpp>

using namespace dealii;
using namespace icepack;

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

  VectorField<2> tau = ssa.driving_stress(s, h);

  return 0;
}
