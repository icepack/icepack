
#include "output_mesh.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <fstream>
#include <string>

using namespace dealii;

// Load in a gmsh grid.
void grid_1 (Triangulation<2> &triangulation,
               const std::string &filename)
{
  GridIn<2> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f(filename);
  gridin.read_msh(f);
}


void distribute_dofs (DoFHandler<2> &dof_handler) {
  static const FE_Q<2> finite_element(1);
  dof_handler.distribute_dofs (finite_element);
  CompressedSparsityPattern compressed_sparsity_pattern(dof_handler.n_dofs(),
                                                        dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, compressed_sparsity_pattern);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from (compressed_sparsity_pattern);

  std::ofstream out ("sparsity_pattern.1");
  sparsity_pattern.print_gnuplot(out);
}


void renumber_dofs (DoFHandler<2> &dof_handler)
{
  DoFRenumbering::Cuthill_McKee (dof_handler);
  CompressedSparsityPattern compressed_sparsity_pattern(dof_handler.n_dofs(),
                                                        dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, compressed_sparsity_pattern);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from (compressed_sparsity_pattern);

  std::ofstream out ("sparsity_pattern.2");
  sparsity_pattern.print_gnuplot (out);
}


int main (int argc, char **argv)
{

  Triangulation<2> triangulation;
  grid_1 (triangulation, "circle.msh");

  DoFHandler<2> dof_handler (triangulation);

  distribute_dofs (dof_handler);
  renumber_dofs (dof_handler);

  mesh_info(triangulation, "grid-1.eps");

  return 0;
}
