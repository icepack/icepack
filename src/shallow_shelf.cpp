
#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>


#include "icepack/elliptic_systems.hpp"
#include "icepack/glacier_models/shallow_shelf.hpp"
#include "icepack/ice_thickness.hpp"
#include "icepack/physical_constants.hpp"


namespace icepack
{
  using namespace dealii;

  using EllipticSystems::cell_to_global;
  using EllipticSystems::fill_cell_rhs;
  using EllipticSystems::stress_strain_tensor;

  const double strain_rate = 0.2;  // 1 / year


  ShallowShelf::ShallowShelf (Triangulation<2>&  _triangulation,
                              const Function<2>& _surface,
                              const Function<2>& _bed,
                              const Function<2>& _temperature,
                              const TensorFunction<1, 2>& _boundary_velocity)
    :
    surface (_surface),
    bed (_bed),
    thickness (IceThickness(surface, bed)),
    temperature (_temperature),
    boundary_velocity (_boundary_velocity),
    triangulation (_triangulation),
    dof_handler (triangulation),
    fe (FE_Q<2>(1), 2),
    quadrature_formula (2),
    face_quadrature_formula (2)
  {}


  ShallowShelf::~ShallowShelf ()
  {
    dof_handler.clear ();
  }


  void ShallowShelf::setup_system (const bool initial_step)
  {
    if (initial_step) {
      dof_handler.distribute_dofs (fe);

      hanging_node_constraints.clear ();
      DoFTools::make_hanging_node_constraints (dof_handler,
                                               hanging_node_constraints);
      hanging_node_constraints.close ();

      solution.reinit (dof_handler.n_dofs());

      // Fill the solution by interpolating from the boundary values
      VectorTools::interpolate
        (dof_handler,
         VectorFunctionFromTensorFunction<2> (boundary_velocity),
         solution);
    }

    sparsity_pattern.reinit (dof_handler.n_dofs(),
                             dof_handler.n_dofs(),
                             dof_handler.max_couplings_between_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);

    hanging_node_constraints.condense (sparsity_pattern);

    sparsity_pattern.compress ();
    system_matrix.reinit (sparsity_pattern);

    velocity_solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
  }


  template <class ConstitutiveTensor>
  void ShallowShelf::assemble_system ()
  {
    system_matrix = 0;
    system_rhs    = 0;

    assemble_matrix<ConstitutiveTensor> ();
    assemble_rhs ();

    hanging_node_constraints.condense (system_matrix);
    hanging_node_constraints.condense (system_rhs);

    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values
      (dof_handler,
       0,
       VectorFunctionFromTensorFunction<2> (boundary_velocity),
       boundary_values);

    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
  }


  template <class ConstitutiveTensor>
  void ShallowShelf::assemble_matrix ()
  {
    FEValues<2> fe_values (fe, quadrature_formula,
                           update_values            | update_gradients |
                           update_quadrature_points | update_JxW_values);
    const FEValuesExtractors::Vector velocities (0);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    std::vector<double> temperature_values (n_q_points);
    std::vector<double> thickness_values (n_q_points);
    std::vector<SymmetricTensor<2, 2>> strain_rate_values (n_q_points);

    ConstitutiveTensor C;

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    // Loop over every cell in the triangulation
    for (auto cell: dof_handler.active_cell_iterators()) {
      cell_matrix = 0;
      fe_values.reinit (cell);

      const std::vector<Point<2>>& quadrature_points
        = fe_values.get_quadrature_points();

      temperature.value_list (quadrature_points, temperature_values);
      thickness.value_list (quadrature_points, thickness_values);
      fe_values[velocities].get_function_symmetric_gradients (solution,
                                                              strain_rate_values);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = fe_values.JxW(q);
        const SymmetricTensor<4, 2> Cq = C(temperature_values[q],
                                           thickness_values[q],
                                           strain_rate_values[q]);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          auto eps_phi_i = fe_values[velocities].symmetric_gradient(i, q);

          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            auto eps_phi_j = fe_values[velocities].symmetric_gradient(j, q);

            cell_matrix(i, j) += (eps_phi_i * Cq * eps_phi_j) * dx;
          }
        }
      }

      cell->get_dof_indices (local_dof_indices);
      cell_to_global (cell_matrix, local_dof_indices, system_matrix);
    } // End of loop over `cell`

  } // End of AssembleMatrix



  void ShallowShelf::assemble_rhs ()
  {
    FEValues<2> fe_values (fe, quadrature_formula,
                           update_values            | update_gradients |
                           update_quadrature_points | update_JxW_values);

    FEFaceValues<2> fe_face_values (fe, face_quadrature_formula,
                                    update_values | update_quadrature_points |
                                    update_normal_vectors | update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    std::vector<double> thickness_values (n_q_points);
    std::vector<double> thickness_face_values (n_face_q_points);
    std::vector<double> surface_values (n_q_points);
    std::vector<double> surface_face_values (n_face_q_points);
    std::vector<Tensor<1, 2>> surface_gradient_values (n_q_points);

    Vector<double> cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    for (auto cell: dof_handler.active_cell_iterators()) {
      cell_rhs = 0;
      fe_values.reinit (cell);

      const auto& quadrature_points = fe_values.get_quadrature_points();

      thickness.value_list(quadrature_points, thickness_values);
      surface.gradient_list(quadrature_points, surface_gradient_values);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const Tensor<1, 2> driving_stress
          = -rho_ice * gravity * thickness_values[q] * surface_gradient_values[q];

        fill_cell_rhs<2> (cell_rhs, driving_stress, fe, fe_values, q, dofs_per_cell);
      }

      for (unsigned int face_number = 0;
           face_number < GeometryInfo<2>::faces_per_cell;
           ++face_number)
        if (cell->face(face_number)->at_boundary()
            and
            cell->face(face_number)->boundary_id() == 1) {
          fe_face_values.reinit (cell, face_number);

          const auto& face_quadrature_points = fe_face_values.get_quadrature_points();
          thickness.value_list(face_quadrature_points, thickness_face_values);
          surface.value_list(face_quadrature_points, surface_face_values);

          for (unsigned int q = 0; q < n_face_q_points; ++q) {
            const double h = thickness_face_values[q];
            const double b = surface_face_values[q] - h;
            const Tensor<1, 2> n = fe_face_values.normal_vector(q);
            const Tensor<1, 2> stress_value
              = 0.5 * gravity * (rho_ice * h * h - rho_water * b * b) * n;

            fill_cell_rhs<2>(cell_rhs, stress_value, fe, fe_face_values, q, dofs_per_cell);
          }
        }

      cell->get_dof_indices (local_dof_indices);
      cell_to_global (cell_rhs, local_dof_indices, system_rhs);
    }
  }


  void ShallowShelf::solve ()
  {
    SolverControl solver_control (1000, 1.0e-12);
    SolverCG<>    cg (solver_control);

    SparseILU<double> preconditioner;
    preconditioner.initialize(system_matrix);

    cg.solve (system_matrix, solution, system_rhs, preconditioner);

    hanging_node_constraints.distribute (solution);
  }


  void ShallowShelf::refine_grid ()
  {
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<2>::estimate (dof_handler,
                                      QGauss<1>(2),
                                      typename FunctionMap<2>::type(),
                                      solution,
                                      estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     0.3, 0.03);

    triangulation.prepare_coarsening_and_refinement ();

    SolutionTransfer<2> solution_transfer (dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement (solution);
    triangulation.execute_coarsening_and_refinement ();

    dof_handler.distribute_dofs(fe);

    // Interpolate the solution on the old mesh to the new mesh
    Vector<double> tmp (dof_handler.n_dofs());
    solution_transfer.interpolate (solution, tmp);
    solution = tmp;

    // Having just refined the mesh and interpolated the old solution, we
    // can adjust any newly added points on the boundary so that the
    // boundary values are exact rather than interpolated from the old ones.
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values
      (dof_handler,
       0,
       VectorFunctionFromTensorFunction<2> (boundary_velocity),
       boundary_values);

    for (const auto& dof_val: boundary_values)
      solution(dof_val.first) = dof_val.second;

    // Reconcile the hanging nodes on the new mesh
    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             hanging_node_constraints);
    hanging_node_constraints.close ();
    hanging_node_constraints.distribute (solution);
    setup_system (false);
  }


  void ShallowShelf::output_results (const unsigned int cycle) const
  {
    std::string filename = "solution-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError());

    filename += ".vtk";
    std::ofstream output (filename.c_str());

    DataOut<2> data_out;
    data_out.attach_dof_handler (dof_handler);

    std::vector<std::string> solution_names;
    solution_names.push_back ("x_velocity");
    solution_names.push_back ("y_velocity");

    data_out.add_data_vector (solution, solution_names);
    data_out.build_patches ();
    data_out.write_vtk (output);
  }


  void ShallowShelf::diagnostic_solve ()
  {
    for (unsigned int cycle = 0; cycle < 3; ++cycle) {
      if (cycle == 0) triangulation.refine_global (2);
      else refine_grid ();

      setup_system (cycle == 0);

      assemble_system<EllipticSystems::LinearSSATensor> ();
      solve ();

      Vector<double> difference(triangulation.n_cells());

      for (unsigned int k = 0; k < 5; ++k) {
        velocity_solution = solution;

        assemble_system<EllipticSystems::SSATensor> ();
        solve ();

        velocity_solution -= solution;
        VectorTools::integrate_difference
          (dof_handler, velocity_solution, ZeroFunction<2>(2),
           difference, quadrature_formula, VectorTools::L2_norm);

        const double error = difference.l2_norm();
        std::cout << error << " ";
      }
      std::cout << std::endl;

      output_results (cycle);
    }
  }

} // End of icepack namespace
