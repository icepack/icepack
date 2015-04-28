
#ifndef SHALLOW_SHELF_HPP
#define SHALLOW_SHELF_HPP


#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>

#include "physical_constants.hpp"
#include "ice_thickness.hpp"


namespace ShallowShelfApproximation
{

  using dealii::Triangulation;
  using dealii::Function;
  using dealii::TensorFunction;
  using dealii::DoFHandler;
  using dealii::FESystem;
  using dealii::ConstraintMatrix;
  using dealii::SparsityPattern;
  using dealii::SparseMatrix;
  using dealii::Vector;


  class ShallowShelf
  {
  public:
    ShallowShelf (Triangulation<2>&  _triangulation,
                  const Function<2>& _surface,
                  const Function<2>& _bed,
                  const TensorFunction<1, 2>& _boundary_velocity);
    ~ShallowShelf ();
    void run ();
    //void output (const std::string& filename);

  private:
    void setup_system (const bool initial_step);
    void assemble_system ();
    void solve ();
    void refine_grid ();
    void output_results (const unsigned int cycle) const;

    const Function<2>& surface;
    const Function<2>& bed;
    const IceThickness thickness;
    const TensorFunction<1, 2>& boundary_velocity;

    Triangulation<2>&  triangulation;
    DoFHandler<2>      dof_handler;

    FESystem<2>        fe;

    ConstraintMatrix     hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;
  };


} // End of ShallowShelf namespace

#endif
