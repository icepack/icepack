
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

    protected:
      SparseMatrix<double> L;
    };



    /**
     * This class contains procedures for regularizing the solution of an
     * inverse problem by penalizing the total variation:
     \f[
     R[u; \alpha] =
     \int_\Omega\left(\sqrt{\alpha^2|\nabla u|^2 + \gamma^2} - \gamma\right)dx
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
      TotalVariation(
        const Discretization<dim>&,
        const double alpha,
        const double gamma
      ) : alpha(alpha), gamma(gamma)
      {}

      /**
       * Compute the total variation of a field.
       */
      double operator()(const Field<dim>& u) const
      {
        const QGauss<dim> quad = u.get_discretization().quad();

        FEValues<dim> fe_values(u.get_fe(), quad, DefaultUpdateFlags::flags);
        const typename Field<dim>::extractor_type ex(0);

        const unsigned int n_q_points = quad.size();
        std::vector<Tensor<1, dim> > du_values(n_q_points);

        double total_variation = 0.0;

        for (auto cell: u.get_dof_handler().active_cell_iterators()) {
          fe_values.reinit(cell);

          fe_values[ex].get_function_gradients(
            u.get_coefficients(), du_values
          );

          for (unsigned int q = 0; q < n_q_points; ++q) {
            const double dx = fe_values.JxW(q);
            const Tensor<1, dim> du = alpha * du_values[q];
            const double dA = std::sqrt(du*du + gamma*gamma) - gamma;
            total_variation += dA * dx;
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
        const auto& discretization = u.get_discretization();
        DualField<dim> div_graph_normal(discretization);

        const auto& fe = u.get_fe();
        const auto& dof_handler = u.get_dof_handler();

        const QGauss<dim> quad = discretization.quad();

        FEValues<dim> fe_values(fe, quad, DefaultUpdateFlags::flags);
        const typename Field<dim>::extractor_type ex(0);

        const unsigned int n_q_points = quad.size();
        const unsigned int dofs_per_cell = fe.dofs_per_cell;

        std::vector<Tensor<1, dim> > du_values(n_q_points);

        Vector<double> cell_div_graph_normal(dofs_per_cell);
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

        for (auto cell: dof_handler.active_cell_iterators()) {
          cell_div_graph_normal = 0;
          fe_values.reinit(cell);

          fe_values[ex].get_function_gradients(u.get_coefficients(), du_values);

          for (unsigned int q = 0; q < n_q_points; ++q) {
            const double dx = fe_values.JxW(q);
            const Tensor<1, dim> du = alpha * du_values[q];
            const double dA = std::sqrt(du*du + gamma*gamma);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const Tensor<1, dim> dphi = fe_values[ex].gradient(i, q);
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

    protected:
      /**
       * Gradient scale for non-dimensionalization
       */
      const double alpha;

      /**
       * Cutoff factor below which the square gradient is penalized
       */
      const double gamma;
    };


  } // End of namespace inverse
} // End of namespace icepack

#endif
