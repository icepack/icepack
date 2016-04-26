
#ifndef ICEPACK_INVERSE_REGULARIZATION_HPP
#define ICEPACK_INVERSE_REGULARIZATION_HPP

#include <icepack/field.hpp>

namespace icepack {
  namespace inverse {

    template <int rank, int dim>
    double square_gradient(const FieldType<rank, dim>& phi)
    {
      const auto& fe = phi.get_fe();

      const QGauss<dim> quad = phi.get_discretization().quad();

      FEValues<dim> fe_values(fe, quad, DefaultUpdateFlags::flags);
      const typename FieldType<rank, dim>::extractor_type ex(0);

      const unsigned int n_q_points = quad.size();

      std::vector<typename FieldType<rank, dim>::gradient_type>
        d_phi_values(n_q_points);

      double integral_grad_square = 0.0;

      for (auto cell: phi.get_dof_handler().active_cell_iterators()) {
        fe_values.reinit(cell);

        fe_values[ex].get_function_gradients(
          phi.get_coefficients(), d_phi_values
        );

        for (unsigned int q = 0; q < n_q_points; ++q) {
          const double dx = fe_values.JxW(q);
          const auto d_phi = d_phi_values[q];
          integral_grad_square += (d_phi * d_phi) * dx;
        }
      }

      return 0.5 * integral_grad_square;
    }


    template <int rank, int dim>
    FieldType<rank, dim, dual> laplacian(const FieldType<rank, dim>& phi)
    {
      const auto& discretization = phi.get_discretization();
      FieldType<rank, dim, dual> grad_square_phi(discretization);

      const auto& fe = phi.get_fe();
      const QGauss<dim> quad = discretization.quad();

      FEValues<dim> fe_values(fe, quad, DefaultUpdateFlags::flags);
      const typename FieldType<rank, dim>::extractor_type ex(0);

      const unsigned int n_q_points = quad.size();
      const unsigned int dofs_per_cell = fe.dofs_per_cell;

      std::vector<typename FieldType<rank, dim>::gradient_type>
        d_phi_values(n_q_points);
      Vector<double> cell_grad_square(dofs_per_cell);
      std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

      for (auto cell: phi.get_dof_handler().active_cell_iterators()) {
        cell_grad_square = 0;
        fe_values.reinit(cell);

        fe_values[ex].get_function_gradients(
          phi.get_coefficients(), d_phi_values
        );

        for (unsigned int q = 0; q < n_q_points; ++q) {
          const double dx = fe_values.JxW(q);
          const auto d_phi = d_phi_values[q];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            cell_grad_square(i) += fe_values[ex].gradient(i, q) * d_phi * dx;
        }

        cell->get_dof_indices(local_dof_indices);
        grad_square_phi.get_constraints().distribute_local_to_global(
          cell_grad_square, local_dof_indices, grad_square_phi.get_coefficients()
        );
      }

      return grad_square_phi;
    }

  } // End of namespace inverse
} // End of namespace icepack

#endif
