
#ifndef ICEPACK_INVERSE_REGULARIZATION_HPP
#define ICEPACK_INVERSE_REGULARIZATION_HPP

#include <deal.II/lac/linear_operator.h>

#include <icepack/field.hpp>
#include <icepack/numerics/linear_solve.hpp>

namespace icepack {
  namespace inverse {

    /**
     * Base class for all regularization methods
     */
    template <int dim>
    class Regularizer
    {
    public:
      /**
       * Compute the cost associated with this regularization method for a
       * field, i.e. how much it deviates from some defintion of smoothness
       */
      virtual double operator()(const Field<dim>& u) const = 0;

      /**
       * Compute the derivative of the regularization functional about some
       * input field; this is a linear operator.
       */
      virtual DualField<dim> derivative(const Field<dim>& u) const = 0;

      /**
       * Given a dual field \f$f\f$ and a field \f$u\f$, compute the Hessian
       * of the regularization functional at \f$u\f$ and use this operator to
       * filter \f$\f$ for smoothness.
       */
      virtual Field<dim>
      filter(const Field<dim>& u, const DualField<dim>& f) const = 0;

      /**
       * Implementations need to be able to override the destructor for any
       * resource management
       */
      virtual ~Regularizer() {}
    };



    /**
     * This class contains procedures for regularizing the solution of an
     * inverse problem by penalizing the square gradient:
     \f[
     R[u; \alpha] = \frac{\alpha^2}{2}\int_\Omega|\nabla u|^2 dx.
     \f]
     * Penalizing the square gradient is equivalent to applying a low-pass
     * filter to the solution with smoothing length \f$\alpha\f$.
     */
    template <int dim>
    class SquareGradient : public Regularizer<dim>
    {
    public:
      SquareGradient(const Discretization<dim>& dsc, const double alpha)
        :
        L(dsc.scalar().get_sparsity())
      {
        const auto& field_dsc = dsc.scalar();

        const dealii::ConstantFunction<2> r(alpha*alpha);
        dealii::MatrixCreator::create_laplace_matrix(
          field_dsc.get_dof_handler(),
          dsc.quad(),
          L,
          &r,
          field_dsc.get_constraints()
        );

        SparseMatrix<double> G(dsc.scalar().get_sparsity());
        G.copy_from(dsc.scalar().get_mass_matrix());
        G.add(1.0, L);
        solver.initialize(G);
      }

      /**
       * Evaluate the integrated square gradient of a field.
       */
      double operator()(const Field<dim>& u) const
      {
        return 0.5 * L.matrix_norm_square(u.get_coefficients());
      }

      /**
       * Compute the derivative of the square gradient of a field, i.e. the
       * Laplace operator applied to the field
       */
      DualField<dim>
      derivative(const Field<dim>& u) const
      {
        DualField<dim> laplacian_u(u.get_discretization());
        L.vmult(laplacian_u.get_coefficients(), u.get_coefficients());
        return laplacian_u;
      }

      /**
       * Compute the field \f$u\f$ such that \f$u^*\f$ is closest to \f$f\f$,
       * subject to a penalty on the square gradient
       */
      Field<dim> filter(const Field<dim>&, const DualField<dim>& f) const
      {
        Field<dim> u(f.get_discretization());
        u.get_coefficients() = f.get_coefficients();
        solver.solve(u.get_coefficients());

        return u;
      }

    protected:
      SparseMatrix<double> L;
      SparseDirectUMFPACK solver;
    };



    /**
     * This class contains procedures for regularizing the solution of an
     * inverse problem by penalizing the total variation:
     \f[
     R[u; \alpha] =
     \int_\Omega\left(\sqrt{\alpha^2|\nabla u|^2 + 1} - 1\right)dx
     \f]
     * Strictly speaking, this is the pseudo-Heuber total variation, which is
     * rounded off in order to make the functional differentiable.
     *
     * The total variation of a function can be visualized as the lateral
     * surface area of its graph. Like the square gradient functional,
     * penalizing the total variation is an effective way to eliminated
     * spurious oscillations in the solution of an inverse problem constrained
     * by noisy data. Unlike low-pass filtering, however, total variation
     * filtering does not remove all steep gradients or jump discontinuities.
     * Instead, it tends to confine these interfaces to as small a perimeter as
     * possible where they do exist.
     */
    template <int dim>
    class TotalVariation : public Regularizer<dim>
    {
    public:
      TotalVariation(const Discretization<dim>&, const double alpha)
        :
        alpha(alpha)
      {}

      /**
       * Compute the total variation of a field.
       */
      double operator()(const Field<dim>& u) const
      {
        using gradient_type = typename Field<dim>::gradient_type;

        const QGauss<dim> quad = u.get_discretization().quad();

        FEValues<dim> fe_values(u.get_fe(), quad, DefaultUpdateFlags::flags);
        const typename Field<dim>::extractor_type ex(0);

        const unsigned int n_q_points = quad.size();
        std::vector<gradient_type> du_values(n_q_points);

        double total_variation = 0.0;

        for (auto cell: u.get_dof_handler().active_cell_iterators()) {
          fe_values.reinit(cell);

          fe_values[ex].get_function_gradients(
            u.get_coefficients(), du_values
          );

          for (unsigned int q = 0; q < n_q_points; ++q) {
            const double dx = fe_values.JxW(q);
            const gradient_type du = alpha * du_values[q];
            const double cell_graph_area = std::sqrt(du*du + 1) - 1;
            total_variation += cell_graph_area * dx;
          }
        }

        return total_variation;
      }


      /**
       * Compute the derivative of the total variation of a field \f$u\f$; the
       * derivative of the total variation is a nonlinear elliptic operator,
       * which is related to the minimal surface equation, applied to \f$u\f$.
       */
      DualField<dim> derivative(const Field<dim>& u) const
      {
        using gradient_type = typename Field<dim>::gradient_type;

        const auto& discretization = u.get_discretization();
        DualField<dim> div_graph_normal(discretization);

        const auto& fe = u.get_fe();
        const auto& dof_handler = u.get_dof_handler();

        const QGauss<dim> quad = discretization.quad();

        FEValues<dim> fe_values(fe, quad, DefaultUpdateFlags::flags);
        const typename Field<dim>::extractor_type ex(0);

        const unsigned int n_q_points = quad.size();
        const unsigned int dofs_per_cell = fe.dofs_per_cell;

        std::vector<gradient_type> du_values(n_q_points);

        Vector<double> cell_div_graph_normal(dofs_per_cell);
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

        for (auto cell: dof_handler.active_cell_iterators()) {
          cell_div_graph_normal = 0;
          fe_values.reinit(cell);

          fe_values[ex].get_function_gradients(
            u.get_coefficients(), du_values
          );

          for (unsigned int q = 0; q < n_q_points; ++q) {
            const double dx = fe_values.JxW(q);
            const gradient_type du = alpha * du_values[q];
            const double dA = std::sqrt(du*du + 1);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const gradient_type dphi = fe_values[ex].gradient(i, q);
              cell_div_graph_normal(i) += alpha * du * dphi / dA * dx;
            }
          }

          cell->get_dof_indices(local_dof_indices);
          u.get_constraints().distribute_local_to_global(
            cell_div_graph_normal,
            local_dof_indices,
            div_graph_normal.get_coefficients()
          );
        }

        return div_graph_normal;
      }


      /**
       * Apply a filter to the dual field \f$f\f$ which matches it as best as
       * possible subject to a constraint on the total variation of the output,
       * which is linearized around an input field \f$u\f$.
       *
       * The Hessian of the total variation is an anisotropic elliptic operator
       * where the anisotropy is aligned with the gradient of the input field
       * \f$u\f$.
       */
      Field<dim> filter(const Field<dim>& u, const DualField<dim>& f) const
      {
        // TODO: use matrix-free method w. multigrid + Chebyshev preconditioner

        using value_type = typename Field<dim>::value_type;
        using gradient_type = typename Field<dim>::gradient_type;

        const auto& discretization = u.get_discretization();
        Field<dim> v(discretization);

        SparseMatrix<double> A(discretization.scalar().get_sparsity());
        A = 0;

        const auto& fe = u.get_fe();
        const auto& dof_handler = u.get_dof_handler();

        const QGauss<dim> quad = discretization.quad();

        FEValues<dim> fe_values(fe, quad, DefaultUpdateFlags::flags);
        const typename Field<dim>::extractor_type ex(0);

        const unsigned int n_q_points = quad.size();
        const unsigned int dofs_per_cell = fe.dofs_per_cell;

        std::vector<gradient_type> du_values(n_q_points);

        dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

        for (auto cell: dof_handler.active_cell_iterators()) {
          cell_matrix = 0;
          fe_values.reinit(cell);

          fe_values.get_function_gradients(u.get_coefficients(), du_values);
          for (unsigned int q = 0; q < n_q_points; ++q) {
            const double dx = fe_values.JxW(q);
            const gradient_type du = alpha * du_values[q];
            const double dA = std::sqrt(du*du + 1);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const auto phi_i = fe_values[ex].value(i, q);
              const auto d_phi_i = fe_values[ex].gradient(i, q);
              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const auto phi_j = fe_values[ex].value(j, q);
                const auto d_phi_j = fe_values[ex].gradient(j, q);

                const double cell_mass = phi_i * phi_j;
                const auto tau = du / dA;
                const double cell_div_graph_normal =
                  (d_phi_i * d_phi_j - (d_phi_i * tau) * (tau * d_phi_j)) / dA;
                cell_matrix(i, j) +=
                  (cell_mass + alpha*alpha * cell_div_graph_normal) * dx;
              }
            }
          }

          cell->get_dof_indices(local_dof_indices);
          u.get_constraints().distribute_local_to_global(
            cell_matrix, local_dof_indices, A
          );
        }

        A.compress(dealii::VectorOperation::add);

        SparseDirectUMFPACK direct_solver;
        direct_solver.initialize(A);

        v.get_coefficients() = f.get_coefficients();
        direct_solver.solve(v.get_coefficients());

        return v;
      }

    protected:
      const double alpha;
    };


  } // End of namespace inverse
} // End of namespace icepack

#endif
