
#include "output_mesh.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_in.h>
#include <fstream>

using namespace dealii;

// Load in a gmsh grid.
void grid_1 ()
{
  Triangulation<2> triangulation;

  GridIn<2> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f("circle.msh");
  gridin.read_msh(f);

  mesh_info(triangulation, "grid-1.eps");

}



int main (int argc, char **argv)
{
  grid_1 ();

  return 0;
}
