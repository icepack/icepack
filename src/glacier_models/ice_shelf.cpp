
#include <deal.II/numerics/matrix_tools.h>

#include <icepack/physics/constants.hpp>
#include <icepack/numerics/linear_solve.hpp>
#include <icepack/glacier_models/ice_shelf.hpp>

namespace icepack {

  using dealii::FullMatrix;

  using dealii::FEValues;
  namespace FEValuesExtractors = dealii::FEValuesExtractors;


  /* ================
   * Helper functions
   * ================ */

  /**
   * Construct the system matrix for the diagnostic equations
   */
  template <Linearity linearity>
  void velocity_matrix(
    SparseMatrix<double>& A,
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u0,
    const IceShelf& ice_shelf
  )
  {
    A = 0;

    const auto& u_fe = u0.get_fe();
    const auto& u_dof_handler = u0.get_dof_handler();

    const QGauss<2>& quad = ice_shelf.get_discretization().quad();

    FEValues<2> u_fe_values(u_fe, quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Vector exv(0);

    FEValues<2> h_fe_values(h.get_fe(), quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Scalar exs(0);

    const unsigned int n_q_points = quad.size();
    const unsigned int dofs_per_cell = u_fe.dofs_per_cell;

    std::vector<double> h_values(n_q_points);
    std::vector<double> theta_values(n_q_points);
    std::vector<SymmetricTensor<2, 2>> strain_rate_values(n_q_points);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    auto cell = u_dof_handler.begin_active();
    auto h_cell = h.get_dof_handler().begin_active();
    for (; cell != u_dof_handler.end(); ++cell, ++h_cell) {
      cell_matrix = 0;
      u_fe_values.reinit(cell);
      h_fe_values.reinit(h_cell);

      h_fe_values[exs].get_function_values(h.get_coefficients(), h_values);
      h_fe_values[exs].get_function_values(theta.get_coefficients(), theta_values);
      u_fe_values[exv].get_function_symmetric_gradients(
        u0.get_coefficients(), strain_rate_values
      );

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = u_fe_values.JxW(q);
        const double H = h_values[q];
        const double T = theta_values[q];
        const SymmetricTensor<2, 2> eps = strain_rate_values[q];
        const SymmetricTensor<4, 2> C =
          ice_shelf.constitutive_tensor.C<linearity>(H, T, eps);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const auto eps_phi_i = u_fe_values[exv].symmetric_gradient(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const auto eps_phi_j = u_fe_values[exv].symmetric_gradient(j, q);
            cell_matrix(i, j) += (eps_phi_i * C * eps_phi_j) * dx;
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      u0.get_constraints().distribute_local_to_global(
        cell_matrix, local_dof_indices, A
      );
    }

    A.compress(dealii::VectorOperation::add);
  }


  /**
   * Solve the diagnostic equations using Newton's method.
   * Note that this requires a sufficiently good initial guess, e.g. obtained
   * from a few iterations of Picard's method.
   */
  VectorField<2> newton_solve(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u0,
    const IceShelf& ice_shelf,
    const double tolerance,
    const unsigned int max_iterations
  )
  {
    const auto& vector_dsc = u0.get_field_discretization();
    SparseMatrix<double> A(vector_dsc.get_sparsity());

    VectorField<2> u;
    u.copy_from(u0);
    auto boundary_values = vector_dsc.zero_boundary_values();

    const VectorField<2> tau = ice_shelf.driving_stress(h);
    const double tau_norm = norm(tau);

    VectorField<2> r = ice_shelf.residual(h, theta, u, tau);
    Vector<double>& R = r.get_coefficients();

    Vector<double>& U = u.get_coefficients();
    Vector<double> dU(U.size());

    double error = 1.0e16;
    for (unsigned int i = 0; i < max_iterations && error > tolerance; ++i) {
      // Fill the system matrix
      velocity_matrix<linearized>(A, h, theta, u, ice_shelf);
      dealii::MatrixTools::apply_boundary_values(boundary_values, A, dU, R, false);

      // Solve the linear system with the updated matrix
      linear_solve(A, dU, R, u.get_constraints());
      U.add(1.0, dU);

      // Compute the relative difference between the new and old solutions
      r = ice_shelf.residual(h, theta, u, tau);
      error = norm(r) / tau_norm;
    }

    return u;
  }


  /**
   * Solve the diagnostic equations using Picard's method.
   */
  VectorField<2> picard_solve(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u0,
    const IceShelf& ice_shelf,
    const double tolerance,
    const unsigned int max_iterations
  )
  {
    const auto& vector_dsc = u0.get_field_discretization();
    SparseMatrix<double> A(vector_dsc.get_sparsity());

    VectorField<2> u, u_old;
    u_old.copy_from(u0);  u.copy_from(u0);
    auto boundary_values = interpolate_boundary_values(u0);

    VectorField<2> tau = ice_shelf.driving_stress(h);
    Vector<double>& F = tau.get_coefficients();
    Vector<double>& U = u.get_coefficients();
    Vector<double>& U_old = u_old.get_coefficients();

    double error = 1.0e16;
    for (unsigned int i = 0; i < max_iterations && error > tolerance; ++i) {
      // Fill the system matrix
      velocity_matrix<nonlinear>(A, h, theta, u, ice_shelf);
      dealii::MatrixTools::apply_boundary_values(boundary_values, A, U, F, false);

      // Solve the linear system with the updated matrix
      linear_solve(A, U, F, u.get_constraints());

      // Compute the relative change in the solution
      error = dist(u, u_old) / norm(u);
      U_old = U;
    }

    return u;
  }


  /* ============================
   * Member functions of IceShelf
   * ============================ */

  IceShelf::IceShelf(const Triangulation<2>& tria, const unsigned int p)
    :
    DepthAveragedModel(tria, p)
  {}


  /*
   * Diagnostic/prognostic model solves
   */

  VectorField<2>
  IceShelf::driving_stress(const Field<2>& h) const
  {
    // Initialize the VectorField for the driving stress
    VectorField<2> tau(discretization);
    const auto& tau_fe = tau.get_fe();
    const auto& tau_dof_handler = tau.get_dof_handler();

    const QGauss<2> quad = discretization.quad();

    FEValues<2> tau_fe_values(tau_fe, quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Vector exv(0);

    FEValues<2> h_fe_values(h.get_fe(), quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Scalar exs(0);

    // Initialize storage for cell-local data
    const unsigned int n_q_points = quad.size();
    const unsigned int dofs_per_cell = tau_fe.dofs_per_cell;

    std::vector<double> h_values(n_q_points);

    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    const double Rho = rho_ice * gravity * (1 - rho_ice / rho_water);

    // Create cell iterators from the tau and h DoFHandlers; these will be
    // iterated jointly.
    auto cell = tau_dof_handler.begin_active();
    auto h_cell = h.get_dof_handler().begin_active();
    for (; cell != tau_dof_handler.end(); ++cell, ++h_cell) {
      cell_rhs = 0;
      tau_fe_values.reinit(cell);
      h_fe_values.reinit(h_cell);

      h_fe_values[exs].get_function_values(h.get_coefficients(), h_values);

      // Add up the driving stress contributions to the cell right-hand side
      // from each quadrature point.
      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = tau_fe_values.JxW(q);
        const double H = h_values[q];

        const double tau_q = 0.5 * Rho * H * H;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          cell_rhs(i) += tau_q * tau_fe_values[exv].divergence(i, q) * dx;
      }

      cell->get_dof_indices(local_dof_indices);
      tau.get_constraints().distribute_local_to_global(
        cell_rhs, local_dof_indices, tau.get_coefficients()
      );
    }

    return tau;
  }


  VectorField<2> IceShelf::residual(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u,
    const VectorField<2>& f
  ) const
  {
    VectorField<2> r;
    r.copy_from(f);

    const auto& u_fe = u.get_fe();
    const auto& u_dof_handler = u.get_dof_handler();

    const QGauss<2>& quad = discretization.quad();

    FEValues<2> u_fe_values(u_fe, quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Vector exv(0);

    FEValues<2> h_fe_values(h.get_fe(), quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Scalar exs(0);

    const unsigned int n_q_points = quad.size();
    const unsigned int dofs_per_cell = u_fe.dofs_per_cell;

    std::vector<double> h_values(n_q_points);
    std::vector<double> theta_values(n_q_points);
    std::vector<SymmetricTensor<2, 2>> strain_rate_values(n_q_points);

    Vector<double> cell_residual(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    auto cell = u_dof_handler.begin_active();
    auto h_cell = h.get_dof_handler().begin_active();
    for (; cell != u_dof_handler.end(); ++cell, ++h_cell) {
      cell_residual = 0;
      u_fe_values.reinit(cell);
      h_fe_values.reinit(h_cell);

      h_fe_values[exs].get_function_values(h.get_coefficients(), h_values);
      h_fe_values[exs].get_function_values(theta.get_coefficients(), theta_values);
      u_fe_values[exv].get_function_symmetric_gradients(
        u.get_coefficients(), strain_rate_values
      );

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = u_fe_values.JxW(q);
        const double H = h_values[q];
        const double T = theta_values[q];
        const SymmetricTensor<2, 2> eps = strain_rate_values[q];

        const SymmetricTensor<4, 2> C =
          constitutive_tensor.C<nonlinear>(H, T, eps);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const auto eps_phi_i = u_fe_values[exv].symmetric_gradient(i, q);
          cell_residual(i) -= (eps_phi_i * C * eps) * dx;
        }
      }

      cell->get_dof_indices(local_dof_indices);
      u.get_constraints().distribute_local_to_global(
        cell_residual, local_dof_indices, r.get_coefficients()
      );
    }


    const unsigned int n_dofs = u_dof_handler.n_dofs();
    std::vector<bool> boundary_dofs(n_dofs);

    // TODO: stop using the magic number 0 for the part of the boundary with
    // Dirichlet conditions; use an enum, preprocessor define, etc. so that
    // it's more obvious what this is.
    const std::set<dealii::types::boundary_id> boundary_ids = {0};
    dealii::DoFTools::extract_boundary_dofs(
      u_dof_handler, dealii::ComponentMask(), boundary_dofs, boundary_ids
    );
    for (unsigned int i = 0; i < n_dofs; ++i)
      if (boundary_dofs[i]) r.get_coefficients()(i) = 0;

    return r;
  }


  VectorField<2> IceShelf::diagnostic_solve(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u0
  ) const
  {
    auto u = picard_solve(h, theta, u0, *this, 0.1, 5);
    return newton_solve(h, theta, u, *this, 1.0e-10, 100);
  }


  VectorField<2> IceShelf::adjoint_solve(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u0,
    const VectorField<2>& rhs
  ) const
  {
    VectorField<2> lambda(discretization);
    const auto& vector_dsc = lambda.get_field_discretization();

    VectorField<2> f;
    f.copy_from(rhs);

    Vector<double>& Lambda = lambda.get_coefficients();
    Vector<double>& F = f.get_coefficients();

    SparseMatrix<double> A(lambda.get_field_discretization().get_sparsity());
    velocity_matrix<linearized>(A, h, theta, u0, *this);
    dealii::MatrixTools::apply_boundary_values(
      vector_dsc.zero_boundary_values(), A, Lambda, F, false
    );

    linear_solve(A, Lambda, F, lambda.get_constraints());

    return lambda;
  }


} // End of icepack namespace
