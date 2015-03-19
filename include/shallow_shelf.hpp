
#ifndef SHALLOW_SHELF_HPP
#define SHALLOW_SHELF_HPP

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

#include <fstream>
#include <iostream>

#include "physical_constants.hpp"

// using dealii::Triangulation;
// using dealii::Function;
using namespace dealii;  // figure out exactly what you need


class ShallowShelfProblem
{
public:
  ShallowShelfProblem (Triangulation<2>& _triangulation,
                       const Function<2>& _bed,
                       const Function<2>& _thickness,
                       const Function<2>& _beta);
  ~ShallowShelfProblem ();
  void run ();
  //void output (const std::string& filename);

private:
  void setup_system ();
  void assemble_system ();
  void solve ();

  const Function<2>& bed;
  const Function<2>& thickness;
  const Function<2>& beta;

  class IceSurface : public Function<2>
  {
  public:
    IceSurface(ShallowShelfProblem& _ssa) : ssa(_ssa) { }

    double value(const Point<2>& x, const unsigned int component) const
    {
      return std::max(ssa.bed.value(x, 0) + ssa.thickness.value(x, 0),
                      (1.0 - rho_ice/rho_water) * ssa.thickness.value(x, 0));
    }

  private:
    ShallowShelfProblem& ssa;
  };

  const IceSurface& surface;

  class DrivingStress : public Function<2>
  {
  public:
    DrivingStress(ShallowShelfProblem& _ssa) : ssa(_ssa) { }

    void vector_value(const Point<2>& x,
                      Vector<double>& values) const
    {
      Tensor<1, 2> grad = ssa.surface.gradient(x, 0);
      // Figure out how deal.ii tensors work...
    }

  private:
    ShallowShelfProblem& ssa;
  };

  const DrivingStress& driving_stress;

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
