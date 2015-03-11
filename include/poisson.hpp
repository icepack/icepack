
#ifndef POISSON_HPP
#define POISSON_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

/* Using declarations, for whenever I decide to abandon the bad practice
   of using the entire dealii namespace
using dealii::Triangulation;
using dealii::Function;
using dealii::FE_Q;
using dealii::FEValues;
using dealii::DoFHandler;
using dealii::DoFTools::make_sparsity_pattern;
using dealii::UpdateFlags;
using dealii::SparsityPattern;
using dealii::CompressedSparsityPattern;
using dealii::SparseMatrix;
using dealii::Vector;
using dealii::QGauss;
using dealii::FullMatrix;
using dealii::SolverCG;
using dealii::PreconditionSSOR;*/
using namespace dealii;

template <int dim>
class PoissonProblem
{
public:
  PoissonProblem(Triangulation<dim>& _triangulation,
                 const Function<dim>& _coefficient,
                 const Function<dim>& _rhs);

  void run();
  void output(const std::string& filename);

private:

  void setup_system();
  void assemble_system();
  void solve();

  Triangulation<dim>&  triangulation;

  const Function<dim>& coefficient;
  const Function<dim>& rhs;

  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
};


template <int dim>
PoissonProblem<dim>::PoissonProblem (Triangulation<dim>& _triangulation,
                                     const Function<dim>& _coefficient,
                                     const Function<dim>& _rhs)
  :
  triangulation (_triangulation),
  coefficient (_coefficient),
  rhs (_rhs),
  fe (1),
  dof_handler (triangulation)
{
}


template <int dim>
void PoissonProblem<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  CompressedSparsityPattern c_sparsity (dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
  sparsity_pattern.copy_from(c_sparsity);

  system_matrix.reinit (sparsity_pattern);
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}


template <int dim>
void PoissonProblem<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(2);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  std::vector<double>    coefficient_values (n_q_points);
  std::vector<double>    rhs_values (n_q_points);

  for (auto cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit (cell);

      coefficient.value_list (fe_values.get_quadrature_points(),
                              coefficient_values);
      rhs.value_list (fe_values.get_quadrature_points(),
                      rhs_values);

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (coefficient_values[q_index] *
                                   fe_values.shape_grad(i, q_index) *
                                   fe_values.shape_grad(j, q_index) *
                                   fe_values.JxW(q_index));

            cell_rhs(i) += (rhs_values[q_index] *
                            fe_values.shape_value(i, q_index) *
                            fe_values.JxW(q_index));
          }


      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }


  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<dim>(),
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);

}


template <int dim>
void PoissonProblem<dim>::solve ()
{
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve (system_matrix, solution, system_rhs,
                preconditioner);
}


template <int dim>
void PoissonProblem<dim>::run()
{
  setup_system ();
  assemble_system ();
  solve ();
}


template <int dim>
void PoissonProblem<dim>::output(const std::string& filename)
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, filename);

  data_out.build_patches ();
  DataOutBase::EpsFlags eps_flags;
  eps_flags.z_scaling = 4;
  eps_flags.azimut_angle = 40;
  eps_flags.turn_angle   = 10;
  data_out.set_flags (eps_flags);

  std::ofstream output ((filename + ".eps").c_str());

  data_out.write_eps (output);
}


#endif
