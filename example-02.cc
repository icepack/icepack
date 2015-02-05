

// See Deal.II Step-3 tutorial


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

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <string>

using namespace dealii;


class Example02
{
public:
  Example02 (const std::string &mesh_file_name);
  void run ();

private:
  void make_grid (const std::string &mesh_file_name);
  void setup_system ();
  void assemble_system ();
  void solve ();
  void output_results () const;

  Triangulation<2> triangulation;
  FE_Q<2> fe;
  DoFHandler<2> dof_handler;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};

Example02::Example02 (const std::string &mesh_file_name)
  :
  fe (1),
  dof_handler(triangulation)
{
  make_grid(mesh_file_name);
}


// Load in a gmsh grid.
void Example02::make_grid(const std::string &mesh_file_name)
{
  GridIn<2> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f(mesh_file_name);
  gridin.read_msh(f);
}


void Example02::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
  sparsity_pattern.copy_from(c_sparsity);

  system_matrix.reinit (sparsity_pattern);
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}


void Example02::assemble_system ()
{
  QGauss<2> quadrature_formula(2);
  FEValues<2> fe_values (fe, quadrature_formula,
                         update_values | update_gradients | update_JxW_values);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  for (auto cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    cell_matrix = 0;
    cell_rhs = 0;

    for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) *
                                fe_values.shape_grad(j, q_index) *
                                fe_values.JxW (q_index));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                        1 *
                        fe_values.JxW (q_index));
    }

    cell->get_dof_indices (local_dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        system_matrix.add (local_dof_indices[i],
                           local_dof_indices[j],
                           cell_matrix(i, j));

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
  }

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            /* This 0 denotes the component of
                                               the boundary. Going to need this
                                               later for different BCs at the
                                               ice front, inflow, side walls. */
                                            0,
                                            ZeroFunction<2>(),
                                            boundary_values);

  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}


void Example02::solve ()
{
  SolverControl solver_control (1000, 1e-12);
  SolverCG<> solver (solver_control);

  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());
}


void Example02::output_results () const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();
  std::ofstream output ("solution.gpl");
  data_out.write_gnuplot (output);
}


void Example02::run ()
{
  setup_system();
  assemble_system();
  solve();
  output_results();
}


int main (int argc, char **argv)
{
  Example02 laplace_problem(argv[1]);
  laplace_problem.run();

  return 0;
}
