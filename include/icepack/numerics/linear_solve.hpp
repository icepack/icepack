
#ifndef ICEPACK_LINEAR_SOLVE_HPP
#define ICEPACK_LINEAR_SOLVE_HPP

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>

namespace icepack {

  using dealii::Vector;
  using dealii::SparseMatrix;
  using dealii::ConstraintMatrix;

  /**
   * Solve a symmetric, positive-definite linear system using sparse ILU and
   * the conjugate gradient method.
   */
  void linear_solve(
    const SparseMatrix<double>& A,
    Vector<double>& u,
    const Vector<double>& f,
    const ConstraintMatrix& constraints
  );

  // TODO: procedure for non-symmetric systems, e.g. implicit time
  // discretization of prognostic equations

  // TODO: procedure to solve saddle point problems

}

#endif
