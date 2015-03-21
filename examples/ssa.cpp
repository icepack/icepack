
#include "shallow_shelf.hpp"

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_generator.h>

using namespace dealii;

Triangulation<2> make_domain()
{
  Triangulation<2> tri;
  GridGenerator::hyper_cube (tri);

  for (auto cell: tri.active_cell_iterators())
  {
    for (unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i)
    {
      Point<2>& v = cell->vertex(i);
      if (std::abs(v(0) - 1.0) < 1e-5)
        v(0) = 500.0;

      if (std::abs(v(1) - 1.0) < 1e-5)
        v(1) = 100.0;
    }
  }

  return tri;
}



int main(int argc, char **argv)
{

  Triangulation<2> tri = make_domain();

  return 0;
}
