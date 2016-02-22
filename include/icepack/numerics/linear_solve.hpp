
#ifndef ICEPACK_LINEAR_SOLVE_HPP
#define ICEPACK_LINEAR_SOLVE_HPP

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/sparse_ilu.h>

namespace icepack {

  using dealii::Vector;
  using dealii::SparseMatrix;
  using dealii::ConstraintMatrix;
  using dealii::SolverControl;
  using dealii::SolverCG;
  using dealii::SolverBicgstab;
  using dealii::SparseILU;

  /**
   * Solve a symmetric, positive-definite linear system using sparse ILU and
   * the conjugate gradient method.
   */
  template <class Solver = SolverCG<> >
  void linear_solve(
    const SparseMatrix<double>& A,
    Vector<double>& u,
    const Vector<double>& f,
    const ConstraintMatrix& constraints
  )
  {
    SolverControl solver_control(1000, 1.0e-10);
    // TODO make choice of logging an argument
    solver_control.log_result(false);
    Solver solver(solver_control);

    SparseILU<double> M;
    M.initialize(A);

    solver.solve(A, u, f, M);
    constraints.distribute(u);
  }

  // TODO: procedure to solve saddle point problems

}

#endif
