
#ifndef SSA_HPP
#define SSA_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

// This again is C++:
#include <fstream>
#include <iostream>


namespace Step8
{
  using namespace dealii;

  class ElasticProblem
  {
  public:
    ElasticProblem (const Function<2>& _right_hand_side);
    ~ElasticProblem ();
    void run ();

  private:
    void setup_system ();
    void assemble_system ();
    void solve ();
    void refine_grid ();
    void output_results (const unsigned int cycle) const;

    const Function<2>& right_hand_side;

    Triangulation<2>   triangulation;
    DoFHandler<2>      dof_handler;

    FESystem<2>        fe;

    ConstraintMatrix     hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;
  };




  ElasticProblem::ElasticProblem (const Function<2>& _right_hand_side)
    :
    right_hand_side(_right_hand_side),
    dof_handler (triangulation),
    fe (FE_Q<2>(1), 2)
  {}



  ElasticProblem::~ElasticProblem ()
  {
    dof_handler.clear ();
  }



  void ElasticProblem::setup_system ()
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


  void ElasticProblem::assemble_system ()
  {
    QGauss<2>  quadrature_formula(2);

    FEValues<2> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<double>   nu_values (n_q_points);
    ConstantFunction<2> nu(1.);

    std::vector<Vector<double> > rhs_values (n_q_points,
                                             Vector<double>(2));


    // Loop over every cell of the mesh
    for (auto cell: dof_handler.active_cell_iterators()) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit (cell);

      // Getting values of coefficients / RHS at the quadrature points
      nu.value_list (fe_values.get_quadrature_points(), nu_values);

      right_hand_side.vector_value_list (fe_values.get_quadrature_points(),
                                         rhs_values);

      for (unsigned int i=0; i<dofs_per_cell; ++i) {
        const unsigned int
          component_i = fe.system_to_component_index(i).first;

        for (unsigned int j=0; j<dofs_per_cell; ++j) {
          const unsigned int
            component_j = fe.system_to_component_index(j).first;

          for (unsigned int q_point=0; q_point<n_q_points; ++q_point) {
            cell_matrix(i,j)
              +=
              // First term is 2 * nu * d_i u_i, d_j v_j)
              //                + (nu * d_i u_j, d_j u_i).
              // <code>shape_grad(i,q_point)</code> returns the
              // gradient of the only nonzero component of the i-th
              // shape function at quadrature point q_point. The
              // component <code>comp(i)</code> of the gradient, which
              // is the derivative of this only nonzero vector
              // component of the i-th shape function with respect to
              // the comp(i)th coordinate is accessed by the appended
              // brackets.
              (
                2 *
                (fe_values.shape_grad(i,q_point)[component_i] *
                 fe_values.shape_grad(j,q_point)[component_j] *
                 nu_values[q_point])
                +
                (fe_values.shape_grad(i,q_point)[component_j] *
                 fe_values.shape_grad(j,q_point)[component_i] *
                 nu_values[q_point])
                +
                // The second term is (nu * nabla u_i, nabla v_j).  We
                // need not access a specific component of the
                // gradient, since we only have to compute the scalar
                // product of the two gradients, of which an
                // overloaded version of the operator* takes care, as
                // in previous examples.
                //
                // Note that by using the ?: operator, we only do this
                // if comp(i) equals comp(j), otherwise a zero is
                // added (which will be optimized away by the
                // compiler).
                ((component_i == component_j) ?
                 (fe_values.shape_grad(i,q_point) *
                  fe_values.shape_grad(j,q_point) *
                  nu_values[q_point])  :
                 0)
                )
              *
              fe_values.JxW(q_point);
          }
        }
      }


      for (unsigned int i=0; i<dofs_per_cell; ++i) {
        const unsigned int
          component_i = fe.system_to_component_index(i).first;

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          cell_rhs(i) += fe_values.shape_value(i,q_point) *
            rhs_values[q_point](component_i) *
            fe_values.JxW(q_point);
      }


      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i) {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));

        system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }

    } // End of loop over `cell`

    hanging_node_constraints.condense (system_matrix);
    hanging_node_constraints.condense (system_rhs);

    // The interpolation of the boundary values needs a small modification:
    // since the solution function is vector-valued, so need to be the
    // boundary values. The <code>ZeroFunction</code> constructor accepts a
    // parameter that tells it that it shall represent a vector valued,
    // constant zero function with that many components. By default, this
    // parameter is equal to one, in which case the <code>ZeroFunction</code>
    // object would represent a scalar function. Since the solution vector has
    // <code>dim</code> components, we need to pass <code>dim</code> as number
    // of components to the zero function as well.
    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<2>(2),
                                              boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
  }



  // @sect4{ElasticProblem::solve}

  // The solver does not care about where the system of equations comes, as
  // long as it stays positive definite and symmetric (which are the
  // requirements for the use of the CG solver), which the system indeed
  // is. Therefore, we need not change anything.
  void ElasticProblem::solve ()
  {
    SolverControl           solver_control (1000, 1e-12);
    SolverCG<>              cg (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve (system_matrix, solution, system_rhs,
              preconditioner);

    hanging_node_constraints.distribute (solution);
  }


  // @sect4{ElasticProblem::refine_grid}

  // The function that does the refinement of the grid is the same as in the
  // step-6 example. The quadrature formula is adapted to the linear elements
  // again. Note that the error estimator by default adds up the estimated
  // obtained from all components of the finite element solution, i.e., it
  // uses the displacement in all directions with the same weight. If we would
  // like the grid to be adapted to the x-displacement only, we could pass the
  // function an additional parameter which tells it to do so and do not
  // consider the displacements in all other directions for the error
  // indicators. However, for the current problem, it seems appropriate to
  // consider all displacement components with equal weight.
  void ElasticProblem::refine_grid ()
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

    triangulation.execute_coarsening_and_refinement ();
  }



  void ElasticProblem::output_results (const unsigned int cycle) const
  {
    std::string filename = "solution-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError());

    filename += ".vtk";
    std::ofstream output (filename.c_str());

    DataOut<2> data_out;
    data_out.attach_dof_handler (dof_handler);



    // As said above, we need a different name for each component of the
    // solution function. To pass one name for each component, a vector of
    // strings is used. Since the number of components is the same as the
    // number of dimensions we are working in, the following
    // <code>switch</code> statement is used.
    //
    // We note that some graphics programs have restriction as to what
    // characters are allowed in the names of variables. The library therefore
    // supports only the minimal subset of these characters that is supported
    // by all programs. Basically, these are letters, numbers, underscores,
    // and some other characters, but in particular no whitespace and
    // minus/hyphen. The library will throw an exception otherwise, at least
    // if in debug mode.
    //
    // After listing the 1d, 2d, and 3d case, it is good style to let the
    // program die if we run upon a case which we did not consider. Remember
    // that the <code>Assert</code> macro generates an exception if the
    // condition in the first parameter is not satisfied. Of course, the
    // condition <code>false</code> can never be satisfied, so the program
    // will always abort whenever it gets to the default statement:
    std::vector<std::string> solution_names;
    switch (2)
      {
      case 1:
        solution_names.push_back ("displacement");
        break;
      case 2:
        solution_names.push_back ("x_displacement");
        solution_names.push_back ("y_displacement");
        break;
      case 3:
        solution_names.push_back ("x_displacement");
        solution_names.push_back ("y_displacement");
        solution_names.push_back ("z_displacement");
        break;
      default:
        Assert (false, ExcNotImplemented());
      }

    // After setting up the names for the different components of the solution
    // vector, we can add the solution vector to the list of data vectors
    // scheduled for output. Note that the following function takes a vector
    // of strings as second argument, whereas the one which we have used in
    // all previous examples accepted a string there. In fact, the latter
    // function is only a shortcut for the function which we call here: it
    // puts the single string that is passed to it into a vector of strings
    // with only one element and forwards that to the other function.
    data_out.add_data_vector (solution, solution_names);
    data_out.build_patches ();
    data_out.write_vtk (output);
  }



  void ElasticProblem::run ()
  {
    for (unsigned int cycle=0; cycle<8; ++cycle)
      {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube (triangulation, -1, 1);
            triangulation.refine_global (2);
          }
        else
          refine_grid ();

        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells()
                  << std::endl;

        setup_system ();

        std::cout << "   Number of degrees of freedom: "
                  << dof_handler.n_dofs()
                  << std::endl;

        assemble_system ();
        solve ();
        output_results (cycle);
      }
  }

} // End of Step8 namespace


#endif
