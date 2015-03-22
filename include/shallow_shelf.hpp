
#ifndef SHALLOW_SHELF_HPP
#define SHALLOW_SHELF_HPP


#include <deal.II/base/function.h>
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


using dealii::Triangulation;
using dealii::Function;
using dealii::DoFHandler;
using dealii::FESystem;
using dealii::ConstraintMatrix;
using dealii::SparsityPattern;
using dealii::SparseMatrix;
using dealii::Vector;


class ShallowShelfProblem
{
public:
  ShallowShelfProblem (Triangulation<2>& _triangulation,
                       const Function<2>& _bed,
                       const Function<2>& _surface,
                       const Function<2>& _beta);
  ~ShallowShelfProblem ();
  void run ();
  //void output (const std::string& filename);

private:
  void setup_system ();
  void assemble_system ();
  void solve ();

  const Function<2>& bed;
  const Function<2>& surface;
  const Function<2>& thickness;
  const Function<2>& driving_stress;
  const Function<2>& beta;

  Triangulation<2>     triangulation;
  DoFHandler<2>        dof_handler;
  FESystem<2>          fe;
  ConstraintMatrix     hanging_node_constraints;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       solution;
  Vector<double>       system_rhs;
};

#endif
