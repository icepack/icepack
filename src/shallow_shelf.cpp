
#include "shallow_shelf.hpp"
#include "ice_surface.hpp"
#include "driving_stress.hpp"
#include "physical_constants.hpp"


constexpr double strain_rate = 100.0;  // m / year
constexpr double nu = 0.5 * pow(A0_cold * exp(-Q_cold / (idealgas * Temp)), -1.0/3)
                          * pow(strain_rate, -2.0/3);


ShallowShelfProblem::ShallowShelfProblem (Triangulation<2>& _triangulation,
                                          const Function<2>& _bed,
                                          const Function<2>& _thickness,
                                          const Function<2>& _beta)
  :
  triangulation (_triangulation),
  dof_handler (triangulation),
  fe (FE_Q<2>(1), 2),
  bed (_bed),
  thickness (_thickness),
  surface( IceSurface(bed, thickness) ),
  driving_stress( DrivingStress(thickness, surface) ),
  beta (_beta)
{}


ShallowShelfProblem::~ShallowShelfProblem ()
{
  dof_handler.clear ();
}


void ShallowShelfProblem::setup_system ()
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


void ShallowShelfProblem::assemble_system ()
{
  QGauss<2> quadrature_formula(2);
  FEValues<2> fe_values (fe, quadrature_formula,
                         update_values            | update_gradients |
                         update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  std::vector<double> nu_values (n_q_points);
  std::vector<double> h_values (n_q_points);
  std::vector<double> beta_values (n_q_points);
  std::vector<Vector<double> > rhs_values (n_q_points,
                                           Vector<double>(2));

  for (auto cell: dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit (cell);

      for (unsigned int i = 0; i < n_q_points; ++i) nu_values[i] = nu;
      thickness.value_list (fe_values.get_quadrature_points(), h_values);
      beta.value_list (fe_values.get_quadrature_points(), beta_values);

      //DrivingStress.vector_value_list (fe_values.get_quadrature_points(),
      //                                 rhs_values);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const unsigned int
          component_i = fe.system_to_component_index(i).first;

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              const unsigned int
              component_j = fe.system_to_component_index(j).first;

              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                  cell_matrix(i, j) +=
                    (
                     (fe_values.shape_grad(i, q_point)[component_i] *
                      fe_values.shape_grad(j, q_point)[component_j])
                     +
                     (fe_values.shape_grad(i, q_point)[component_j] *
                      fe_values.shape_grad(j, q_point)[component_i])
                     +
                     2 * ((component_i == component_j) ?
                          (fe_values.shape_grad(i, q_point) *
                           fe_values.shape_grad(j, q_point)) : 0)
                     * nu_values[q_point]
                     * h_values[q_point]
                    ) * fe_values.JxW(q_point);
                } // End of loop over q_point
            } // End of loop over j
        } // End of loop over i

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int
              component_i = fe.system_to_component_index(i).first;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                cell_rhs(i) += fe_values.shape_value(i, q_point) *
                               rhs_values[q_point](component_i) *
                               fe_values.JxW(q_point);
              }
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

    } // End of loop over cell

  hanging_node_constraints.condense (system_matrix);
  hanging_node_constraints.condense (system_rhs);

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<2>(2), // Need right BVs
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);

} // End of AssembleSystem


void ShallowShelfProblem::solve ()
{
  SolverControl solver_control (1000, 1.0e-12);
  SolverCG<>    cg (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  cg.solve (system_matrix, solution, system_rhs,
            preconditioner);

  hanging_node_constraints.distribute (solution);
}


void ShallowShelfProblem::run ()
{
  setup_system ();
  solve ();
}
