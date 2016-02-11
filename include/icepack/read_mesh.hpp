
#ifndef READ_GRID_HPP
#define READ_GRID_HPP

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>

#include <fstream>

namespace icepack {
  using dealii::Triangulation;
  using dealii::GridIn;

  template <int dim>
  Triangulation<dim> read_gmsh_grid(const std::string& mesh_filename)
  {
    GridIn<dim> grid_in;
    Triangulation<dim> triangulation;
    grid_in.attach_triangulation(triangulation);
    std::ifstream file_stream(mesh_filename);
    grid_in.read_msh(file_stream);
    return triangulation;
  }
}

#endif
