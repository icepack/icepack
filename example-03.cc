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
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/grid/tria_boundary_lib.h>

#include <fstream>
#include <sstream>


using namespace dealii;


template <int dim>
class Step5
{
public:
  Step5 (const std::string& mesh_filename);
  void run ();

private:
  void make_grid (const std::string& mesh_filename);
  void setup_system ();
  void assemble_system ();
  void solve ();
  void output_results (const unsigned int cycle) const;

  Triangulation<dim>   triangulation;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
};



template <int dim>
class Coefficient : public Function<dim>
{
public:
  Coefficient ()  : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0) const;
};



template <int dim>
double Coefficient<dim>::value (const Point<dim> &p,
                                const unsigned int /*component*/) const
{
  if (p.square() < 0.5*0.5)
    return 20;
  else
    return 1;
}



template <int dim>
void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
                                   std::vector<double>            &values,
                                   const unsigned int              component) const
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));
  Assert (component == 0,
          ExcIndexRange (component, 0, 1));

  const unsigned int n_points = points.size();

  for (unsigned int i=0; i<n_points; ++i)
    {
      if (points[i].square() < 0.5*0.5)
        values[i] = 20;
      else
        values[i] = 1;
    }
}


template <int dim>
Step5<dim>::Step5 (const std::string& mesh_filename) :
  fe (1),
  dof_handler (triangulation)
{
  make_grid(mesh_filename);
}


// Load in a gmsh grid
template <int dim>
void Step5<dim>::make_grid(const std::string& mesh_filename)
{
  GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f(mesh_filename);
  gridin.read_msh(f);
}


template <int dim>
void Step5<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
  sparsity_pattern.copy_from(c_sparsity);

  system_matrix.reinit (sparsity_pattern);
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}


template <int dim>
void Step5<dim>::assemble_system ()
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

  const Coefficient<dim> coefficient;
  std::vector<double>    coefficient_values (n_q_points);

  for (auto cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit (cell);

      coefficient.value_list (fe_values.get_quadrature_points(),
                              coefficient_values);

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (coefficient_values[q_index] *
                                   fe_values.shape_grad(i,q_index) *
                                   fe_values.shape_grad(j,q_index) *
                                   fe_values.JxW(q_index));

            cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                            1.0 *
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
void Step5<dim>::solve ()
{
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve (system_matrix, solution, system_rhs,
                preconditioner);

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;
}


// @sect4{Step5::output_results and setting output flags}

// Writing output to a file is mostly the same as for the previous example,
// but here we will show how to modify some output options and how to
// construct a different filename for each refinement cycle.
template <int dim>
void Step5<dim>::output_results (const unsigned int cycle) const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");

  data_out.build_patches ();

  // For this example, we would like to write the output directly to a file in
  // Encapsulated Postscript (EPS) format. The library supports this, but
  // things may be a bit more difficult sometimes, since EPS is a printing
  // format, unlike most other supported formats which serve as input for
  // graphical tools. Therefore, you can't scale or rotate the image after it
  // has been written to disk, and you have to decide about the viewpoint or
  // the scaling in advance.
  //
  // The defaults in the library are usually quite reasonable, and regarding
  // viewpoint and scaling they coincide with the defaults of
  // Gnuplot. However, since this is a tutorial, we will demonstrate how to
  // change them. For this, we first have to generate an object describing the
  // flags for EPS output (similar flag classes exist for all supported output
  // formats):
  DataOutBase::EpsFlags eps_flags;
  // They are initialized with the default values, so we only have to change
  // those that we don't like. For example, we would like to scale the z-axis
  // differently (stretch each data point in z-direction by a factor of four):
  eps_flags.z_scaling = 4;
  // Then we would also like to alter the viewpoint from which we look at the
  // solution surface. The default is at an angle of 60 degrees down from the
  // vertical axis, and 30 degrees rotated against it in mathematical positive
  // sense. We raise our viewpoint a bit and look more along the y-axis:
  eps_flags.azimut_angle = 40;
  eps_flags.turn_angle   = 10;
  // That shall suffice. There are more flags, for example whether to draw the
  // mesh lines, which data vectors to use for colorization of the interior of
  // the cells, and so on. You may want to take a look at the documentation of
  // the EpsFlags structure to get an overview of what is possible.
  //
  // The only thing still to be done, is to tell the output object to use
  // these flags:
  data_out.set_flags (eps_flags);
  // The above way to modify flags requires recompilation each time we would
  // like to use different flags. This is inconvenient, and we will see more
  // advanced ways in step-19 where the output flags are determined at run
  // time using an input file (step-19 doesn't show many other things; you
  // should feel free to read over it even if you haven't done step-6 to
  // step-18 yet).

  // Finally, we need the filename to which the results are to be written. We
  // would like to have it of the form <code>solution-N.eps</code>, where N is
  // the number of the refinement cycle. Thus, we have to convert an integer
  // to a part of a string; this can be done using the <code>sprintf</code>
  // function, but in C++ there is a more elegant way: write everything into a
  // special stream (just like writing into a file or to the screen) and
  // retrieve what you wrote as a string. This applies the usual conversions
  // from integer to strings, and one could as well use stream modifiers such
  // as <code>setw</code>, <code>setprecision</code>, and so on. In C++, you
  // can do this by using the so-called stringstream classes:
  std::ostringstream filename;

  // In order to now actually generate a filename, we fill the stringstream
  // variable with the base of the filename, then the number part, and finally
  // the suffix indicating the file type:
  filename << "solution-"
           << cycle
           << ".eps";

  // We can get whatever we wrote to the stream using the <code>str()</code>
  // function. The result is a string which we have to convert to a char*
  // using the <code>c_str()</code> function. Use that as filename for the
  // output stream and then write the data to the file:
  std::ofstream output (filename.str().c_str());

  data_out.write_eps (output);
}


template <int dim>
void Step5<dim>::run ()
{
  for (unsigned int cycle=0; cycle<3; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle != 0)
        triangulation.refine_global (1);

      std::cout << "   Number of active cells: "
                << triangulation.n_active_cells()
                << std::endl
                << "   Total number of cells: "
                << triangulation.n_cells()
                << std::endl;

      setup_system ();
      assemble_system ();
      solve ();
      output_results (cycle);
    }
}



int main (int argc, char **argv)
{
  deallog.depth_console (0);

  Step5<2> laplace_problem_2d(argv[1]);
  laplace_problem_2d.run ();

  return 0;
}
