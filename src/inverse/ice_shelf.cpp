
#include <deal.II/grid/grid_tools.h>

#include <icepack/physics/viscosity.hpp>
#include <icepack/glacier_models/ice_shelf.hpp>
#include <icepack/numerics/linear_solve.hpp>
#include <icepack/inverse/mean_square_error.hpp>

using dealii::SparseMatrix;
namespace FEValuesExtractors = dealii::FEValuesExtractors;

namespace icepack {
  namespace inverse {

    namespace {

      // Compute the product of the mass matrix and the gradient of the
      // objective functional

      Field<2> M_gradient(
        const IceShelf& ice_shelf,
        const Field<2>& h,
        const Field<2>& theta,
        const VectorField<2>& u,
        const VectorField<2>& lambda
      )
      {
        const auto& discretization = ice_shelf.get_discretization();
        Field<2> MdJ(discretization);

        const auto& s_fe = MdJ.get_fe();
        const auto& s_dof_handler = MdJ.get_dof_handler();

        const QGauss<2> quad = discretization.quad();

        FEValues<2> s_fe_values(s_fe, quad, DefaultUpdateFlags::flags);
        const FEValuesExtractors::Scalar exs(0);

        FEValues<2> u_fe_values(u.get_fe(), quad, DefaultUpdateFlags::flags);
        const FEValuesExtractors::Vector exv(0);

        const unsigned int n_q_points = quad.size();
        const unsigned int dofs_per_cell = s_fe.dofs_per_cell;

        std::vector<double> h_values(n_q_points);
        std::vector<double> theta_values(n_q_points);
        std::vector<SymmetricTensor<2, 2>>
          eps_u_values(n_q_points), eps_lambda_values(n_q_points);

        Vector<double> cell_dJ(dofs_per_cell);
        std::vector<dealii::types::global_dof_index>
          local_dof_indices(dofs_per_cell);

        auto cell = s_dof_handler.begin_active();
        auto u_cell = u.get_dof_handler().begin_active();
        for (; cell != s_dof_handler.end(); ++cell, ++u_cell) {
          cell_dJ = 0;
          s_fe_values.reinit(cell);
          u_fe_values.reinit(u_cell);

          s_fe_values[exs].get_function_values(theta.get_coefficients(), theta_values);
          s_fe_values[exs].get_function_values(h.get_coefficients(), h_values);
          u_fe_values[exv].get_function_symmetric_gradients(
            u.get_coefficients(), eps_u_values
          );
          u_fe_values[exv].get_function_symmetric_gradients(
            lambda.get_coefficients(), eps_lambda_values
          );

          for (unsigned int q = 0; q < n_q_points; ++q) {
            const double dx = s_fe_values.JxW(q);
            const double H = h_values[q];
            const double Theta = theta_values[q];
            const SymmetricTensor<2, 2> eps_u = eps_u_values[q];
            const SymmetricTensor<2, 2> eps_lambda = eps_lambda_values[q];
            const double u_dot_lambda =
              eps_u * eps_lambda + trace(eps_u) * trace(eps_lambda);
            const double dB = ice_shelf.constitutive_tensor.rheology.dB(Theta);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const auto phi_i = s_fe_values[exs].value(i, q);
              cell_dJ(i) += 2 * dB * H * u_dot_lambda * phi_i * dx;
            }
          }

          cell->get_dof_indices(local_dof_indices);
          MdJ.get_constraints().distribute_local_to_global(
            cell_dJ, local_dof_indices, MdJ.get_coefficients()
          );
        }

        return MdJ;
      }

    } // End of anonymous namespace



    Field<2> gradient(
      const IceShelf& ice_shelf,
      const Field<2>& h,
      const Field<2>& theta,
      const VectorField<2>& u0,
      const Field<2>& sigma
    )
    {
      const VectorField<2> u = ice_shelf.diagnostic_solve(h, theta, u0);
      const VectorField<2> du = misfit(u, u0, sigma);
      const VectorField<2> lambda = ice_shelf.adjoint_solve(h, theta, u, du);

      Field<2> MdJ = M_gradient(ice_shelf, h, theta, u, lambda);

      const auto& scalar_dsc = ice_shelf.get_discretization().scalar();
      const SparseMatrix<double>& M = scalar_dsc.get_mass_matrix();

      Field<2> dJ(MdJ);

      linear_solve(
        M,
        dJ.get_coefficients(),
        MdJ.get_coefficients(),
        scalar_dsc.get_constraints()
      );

      return dJ;
    }

  }
}
