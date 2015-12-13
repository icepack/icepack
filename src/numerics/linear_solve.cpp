
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_ilu.h>

#include <icepack/numerics/linear_solve.hpp>

namespace icepack {

  using dealii::SolverControl;
  using dealii::SolverCG;
  using dealii::SparseILU;

  void linear_solve(
    const SparseMatrix<double>& A,
    Vector<double>& u,
    const Vector<double>& f,
    const ConstraintMatrix& constraints
  )
  {
    SolverControl solver_control(1000, 1.0e-12);
    solver_control.log_result(false); // silence solver progress output
    SolverCG<> cg(solver_control);

    SparseILU<double> M;
    M.initialize(A);

    cg.solve(A, u, f, M);

    constraints.distribute(u);
  }

}
