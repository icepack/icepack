
#include "shallow_shelf.hpp"


ShallowShelfProblem::ShallowShelfProblem(Triangulation<2>& _triangulation,
                                         const Function<2>& _surface,
                                         const Function<2>& _bed,
                                         const Function<2>& _beta)
  :
  triangulation (_triangulation),
  surface (_surface),
  bed (_bed),
  beta (_beta),
  dof_handler (triangulation),
  fe (FE_Q<2>(1), dim)
{}


void ShallowShelfProblem::setup_system()
{
  dof_handler.distribute_dofs (fe);
  hanging_node_constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           hanging_node_constraints);
  hanging_node_constraints.close ();
  sparsity_pattern.reinit (dof_handler.n_dofs(),
                           dof_handler.n_dofs(),
                           dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
  
  hanging_node_constraints.condense (sparsity_pattern);
  
  sparsity_pattern.compress();

  system_matrix.reinit (sparsity_pattern);
  
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());  
}
