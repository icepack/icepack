
#ifndef ICEPACK_INVERSE_REGULARIZATION_HPP
#define ICEPACK_INVERSE_REGULARIZATION_HPP

#include <icepack/field.hpp>
#include <icepack/numerics/linear_solve.hpp>

namespace icepack {
  namespace inverse {

    template <int rank, int dim>
    double mean_square_gradient(const FieldType<rank, dim>& phi)
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

      const double area = dealii::GridTools::volume(phi.get_triangulation());
      return 0.5 * integral_grad_square / area;
    }


    template <int rank, int dim>
    FieldType<rank, dim> laplacian(const FieldType<rank, dim>& phi)
    {
      const auto& discretization = phi.get_discretization();
      FieldType<rank, dim> grad_square_phi(discretization);

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

      // This is probably a terrible idea. Figure out some way to apply the
      // spatial filter directly to the gradient.
      Vector<double> F(grad_square_phi.get_coefficients());
      SolverControl solver_control(1000, 1.0e-10);
      solver_control.log_result(false);
      SolverCG<> solver(solver_control);

      auto& DPhi = grad_square_phi.get_coefficients();
      const auto& M = phi.get_field_discretization().get_mass_matrix();
      solver.solve(M, DPhi, F, dealii::PreconditionIdentity());
      grad_square_phi.get_constraints().distribute(DPhi);

      const double area = dealii::GridTools::volume(phi.get_triangulation());
      return grad_square_phi / area;
    }

  } // End of namespace inverse
} // End of namespace icepack

#endif
