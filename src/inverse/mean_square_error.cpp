
#include <deal.II/grid/grid_tools.h>

#include <icepack/field.hpp>

using dealii::GridTools::volume;
namespace FEValuesExtractors = dealii::FEValuesExtractors;

namespace icepack {
  namespace inverse {

    double mean_square_error(
      const VectorField<2>& u_model,
      const VectorField<2>& u_observed,
      const Field<2>& sigma
    )
    {
      double mse = 0.0;

      const auto& u_fe = u_model.get_fe();
      const auto& u_dof_handler = u_model.get_dof_handler();
      const auto& discretization = u_model.get_discretization();

      const QGauss<2> quad = discretization.quad();

      FEValues<2> u_fe_values(u_fe, quad, DefaultUpdateFlags::flags);
      const FEValuesExtractors::Vector exv(0);

      FEValues<2> s_fe_values(sigma.get_fe(), quad, DefaultUpdateFlags::flags);
      const FEValuesExtractors::Scalar exs(0);

      const unsigned int n_q_points = quad.size();

      std::vector<Tensor<1, 2>> u_values(n_q_points), uo_values(n_q_points);
      std::vector<double> sigma_values(n_q_points);

      auto cell = u_dof_handler.begin_active();
      auto s_cell = sigma.get_dof_handler().begin_active();
      for (; cell != u_dof_handler.end(); ++cell, ++s_cell) {
        u_fe_values.reinit(cell);
        s_fe_values.reinit(s_cell);

        s_fe_values[exs].get_function_values(sigma.get_coefficients(), sigma_values);
        u_fe_values[exv].get_function_values(u_model.get_coefficients(), u_values);
        u_fe_values[exv].get_function_values(u_observed.get_coefficients(), uo_values);

        for (unsigned int q = 0; q < n_q_points; ++q) {
          const double dx = u_fe_values.JxW(q);
          const double Sigma = sigma_values[q];
          const Tensor<1, 2> dU = u_values[q] - uo_values[q];

          mse += (dU * dU) / (2 * Sigma * Sigma) * dx;
        }
      }

      const double area = volume(discretization.get_triangulation());
      return mse / area;
    }


    VectorField<2> misfit(
      const VectorField<2>& u_model,
      const VectorField<2>& u_observed,
      const Field<2>& sigma
    )
    {
      const auto& u_fe = u_model.get_fe();
      const auto& u_dof_handler = u_model.get_dof_handler();
      const auto& discretization = u_model.get_discretization();

      VectorField<2> du(discretization);

      const QGauss<2>& quad = discretization.quad();

      FEValues<2> s_fe_values(sigma.get_fe(), quad, DefaultUpdateFlags::flags);
      const FEValuesExtractors::Scalar exs(0);

      FEValues<2> u_fe_values(u_fe, quad, DefaultUpdateFlags::flags);
      const FEValuesExtractors::Vector exv(0);

      const unsigned int n_q_points = quad.size();
      const unsigned int dofs_per_cell = u_fe.dofs_per_cell;
      std::vector<Tensor<1, 2>> u_values(n_q_points), uo_values(n_q_points);
      std::vector<double> sigma_values(n_q_points);

      Vector<double> cell_du(dofs_per_cell);
      std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

      auto cell = u_dof_handler.begin_active();
      auto s_cell = sigma.get_dof_handler().begin_active();
      for (; cell != u_dof_handler.end(); ++cell, ++s_cell) {
        cell_du = 0;
        u_fe_values.reinit(cell);
        s_fe_values.reinit(s_cell);

        s_fe_values[exs].get_function_values(
          sigma.get_coefficients(), sigma_values
        );
        u_fe_values[exv].get_function_values(u_model.get_coefficients(), u_values);
        u_fe_values[exv].get_function_values(u_observed.get_coefficients(), uo_values);

        for (unsigned int q = 0; q < n_q_points; ++q) {
          const double dx = u_fe_values.JxW(q);
          const Tensor<1, 2> U = u_values[q];
          const Tensor<1, 2> U0 = uo_values[q];
          const double Sigma = sigma_values[q];
          const Tensor<1, 2> dU = (U - U0) / (Sigma * Sigma);

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            const Tensor<1, 2> phi_i = u_fe_values[exv].value(i, q);
            cell_du(i) += dU * phi_i * dx;
          }
        }

        cell->get_dof_indices(local_dof_indices);
        du.get_constraints().distribute_local_to_global(
          cell_du, local_dof_indices, du.get_coefficients()
        );
      }

      const double area = volume(discretization.get_triangulation());
      return du / area;
    }


  } // End of namespace inverse
} // End of namespace icepack
