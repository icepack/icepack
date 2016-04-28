
#ifndef ICEPACK_INVERSE_REGULARIZATION_HPP
#define ICEPACK_INVERSE_REGULARIZATION_HPP

#include <deal.II/lac/linear_operator.h>

#include <icepack/field.hpp>
#include <icepack/numerics/linear_solve.hpp>

namespace icepack {
  namespace inverse {

    using dealii::linear_operator;

    template <int rank, int dim>
    class SquareGradient
    {
    public:
      SquareGradient(const Discretization<dim>& dsc, const double alpha)
        :
        L(get<rank>(dsc).get_sparsity()),
        M(&get<rank>(dsc).get_mass_matrix())
      {
        const auto& field_dsc = get<rank>(dsc);

        dealii::ConstantFunction<dim> Alpha2(alpha * alpha);

        dealii::MatrixCreator::create_laplace_matrix(
          field_dsc.get_dof_handler(),
          dsc.quad(),
          L,
          &Alpha2,
          field_dsc.get_constraints()
        );
      }

      double operator()(const FieldType<rank, dim>& phi) const
      {
        return 0.5 * L.matrix_norm_square(phi.get_coefficients());
      }

      FieldType<rank, dim, dual> derivative(const FieldType<rank, dim>& phi) const
      {
        FieldType<rank, dim, dual> laplacian_phi(transpose(phi));
        L.vmult(laplacian_phi.get_coefficients(), phi.get_coefficients());
        return laplacian_phi;
      }

      FieldType<rank, dim> filter(
        const FieldType<rank, dim, primal>&,
        const FieldType<rank, dim, dual>& f
      ) const
      {
        FieldType<rank, dim> psi(f.get_discretization());

        const auto A = linear_operator(*M) + linear_operator(L);

        SolverControl solver_control(1000, 1.0e-10);
        SolverCG<> solver(solver_control);

        // TODO: use an actual preconditioner
        dealii::PreconditionIdentity P;
        solver.solve(A, psi.get_coefficients(), f.get_coefficients(), P);

        return psi;
      }

    protected:
      SparseMatrix<double> L;
      SmartPointer<const SparseMatrix<double> > M;
    };

  } // End of namespace inverse
} // End of namespace icepack

#endif
