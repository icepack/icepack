
#include <deal.II/numerics/matrix_tools.h>

#include <icepack/physics/constants.hpp>
#include <icepack/numerics/linear_solve.hpp>
#include <icepack/glacier_models/ice_shelf.hpp>

namespace icepack {

  using dealii::FullMatrix;

  using dealii::FEValues;
  namespace FEValuesExtractors = dealii::FEValuesExtractors;
  namespace MatrixTools = dealii::MatrixTools;


  /* ================
   * Helper functions
   * ================ */

  /**
   * Construct the system matrix for the diagnostic equations
   */
  template <Linearity linearity>
  SparseMatrix<double> velocity_matrix(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u0,
    const IceShelf& ice_shelf
  )
  {
    const auto& discretization = ice_shelf.get_discretization();

    SparseMatrix<double> A(discretization.vector().get_sparsity());
    A = 0;

    const FiniteElement<2>& u_fe = u0.get_fe();
    const QGauss<2> quad = discretization.quad();

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
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    for (const auto& it: discretization) {
      cell_matrix = 0;
      u_fe_values.reinit(discretization.vector_cell_iterator(it));
      h_fe_values.reinit(discretization.scalar_cell_iterator(it));

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

      discretization.vector_cell_iterator(it)->get_dof_indices(local_dof_ids);
      u0.get_constraints().distribute_local_to_global(
        cell_matrix, local_dof_ids, A
      );
    }

    A.compress(dealii::VectorOperation::add);

    return A;
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

    VectorField<2> u(u0);
    auto boundary_values = vector_dsc.zero_boundary_values();

    const DualVectorField<2> tau = ice_shelf.driving_stress(h);
    const double tau_norm = norm(tau);

    DualVectorField<2> r = ice_shelf.derivative(h, theta, u);
    Vector<double>& R = r.get_coefficients();

    Vector<double>& U = u.get_coefficients();
    Vector<double> dU(U.size());

    double error = std::numeric_limits<double>::infinity();
    for (unsigned int i = 0; i < max_iterations && error > tolerance; ++i) {
      // Fill the system matrix
      auto A = ice_shelf.hessian(h, theta, u);
      MatrixTools::apply_boundary_values(boundary_values, A, dU, R, false);

      // Solve the linear system with the updated matrix
      linear_solve(A, dU, R, u.get_constraints());
      U.add(-1.0, dU);

      // Compute the relative difference between the new and old solutions
      r = ice_shelf.derivative(h, theta, u);
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
    VectorField<2> u(u0), u_old(u0);
    auto boundary_values = interpolate_boundary_values(u0);

    DualVectorField<2> tau = ice_shelf.driving_stress(h);
    Vector<double>& F = tau.get_coefficients();
    Vector<double>& U = u.get_coefficients();
    Vector<double>& U_old = u_old.get_coefficients();

    double error = std::numeric_limits<double>::infinity();
    for (unsigned int i = 0; i < max_iterations && error > tolerance; ++i) {
      // Fill the system matrix
      auto A = velocity_matrix<nonlinear>(h, theta, u, ice_shelf);
      MatrixTools::apply_boundary_values(boundary_values, A, U, F, false);

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

  /*
   * Diagnostic/prognostic model solves
   */

  DualVectorField<2> IceShelf::driving_stress(const Field<2>& h) const
  {
    // Initialize the VectorField for the driving stress
    DualVectorField<2> tau(discretization);

    const FiniteElement<2>& tau_fe = tau.get_fe();
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
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    const double Rho = rho_ice * gravity * (1 - rho_ice / rho_water);

    for (const auto& it: discretization) {
      cell_rhs = 0;
      tau_fe_values.reinit(discretization.vector_cell_iterator(it));
      h_fe_values.reinit(discretization.scalar_cell_iterator(it));

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

      discretization.vector_cell_iterator(it)->get_dof_indices(local_dof_ids);
      tau.get_constraints().distribute_local_to_global(
        cell_rhs, local_dof_ids, tau.get_coefficients()
      );
    }

    return tau;
  }


  double IceShelf::action(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u
  ) const
  {
    double P = 0.0;

    const QGauss<2>& quad = discretization.quad();
    const unsigned int n_q_points = quad.size();

    FEValues<2> u_fe_values(u.get_fe(), quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Vector exv(0);

    FEValues<2> h_fe_values(h.get_fe(), quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Scalar exs(0);

    std::vector<double> h_values(n_q_points);
    std::vector<double> theta_values(n_q_points);
    std::vector<SymmetricTensor<2, 2>> eps_values(n_q_points);

    const double Rho = rho_ice * (1 - rho_ice / rho_water);
    const double n = constitutive_tensor.rheology.n;

    for (const auto& it: discretization) {
      u_fe_values.reinit(discretization.vector_cell_iterator(it));
      h_fe_values.reinit(discretization.scalar_cell_iterator(it));

      h_fe_values[exs].get_function_values(h.get_coefficients(), h_values);
      h_fe_values[exs].get_function_values(theta.get_coefficients(), theta_values);
      u_fe_values[exv].get_function_symmetric_gradients(
        u.get_coefficients(), eps_values
      );

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = u_fe_values.JxW(q);
        const double H = h_values[q];
        const double T = theta_values[q];
        const SymmetricTensor<2, 2> eps = eps_values[q];
        const SymmetricTensor<4, 2> C =
          constitutive_tensor.C<nonlinear>(H, T, eps);

        const double strain_power = n/(n + 1) * eps * C * eps;
        const double driving_stress_power = -Rho * gravity * H*H * trace(eps)/2;

        P += (strain_power + driving_stress_power) * dx;
      }
    }

    return P;
  }


  DualVectorField<2> IceShelf::derivative(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u
  ) const
  {
    DualVectorField<2> r = -driving_stress(h);

    const FiniteElement<2>& u_fe = u.get_fe();
    const QGauss<2> quad = discretization.quad();

    FEValues<2> u_fe_values(u_fe, quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Vector exv(0);

    FEValues<2> h_fe_values(h.get_fe(), quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Scalar exs(0);

    const unsigned int n_q_points = quad.size();
    const unsigned int dofs_per_cell = u_fe.dofs_per_cell;

    std::vector<double> h_values(n_q_points);
    std::vector<double> theta_values(n_q_points);
    std::vector<SymmetricTensor<2, 2>> strain_rate_values(n_q_points);

    Vector<double> cell_derivative(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_ids(dofs_per_cell);

    for (const auto& it: discretization) {
      cell_derivative = 0;
      u_fe_values.reinit(discretization.vector_cell_iterator(it));
      h_fe_values.reinit(discretization.scalar_cell_iterator(it));

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
          cell_derivative(i) += (eps_phi_i * C * eps) * dx;
        }
      }

      discretization.vector_cell_iterator(it)->get_dof_indices(local_dof_ids);
      r.get_constraints().distribute_local_to_global(
        cell_derivative, local_dof_ids, r.get_coefficients()
      );
    }

    const DoFHandler<2>& u_dof_handler = u.get_dof_handler();
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


  SparseMatrix<double> IceShelf::hessian(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u
  ) const
  {
    return velocity_matrix<linearized>(h, theta, u, *this);
  }


  VectorField<2> IceShelf::diagnostic_solve(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u0
  ) const
  {
    auto u = picard_solve(h, theta, u0, *this, picard_tolerance, max_iterations/4);
    return newton_solve(h, theta, u, *this, newton_tolerance, max_iterations);
  }


  VectorField<2> IceShelf::adjoint_solve(
    const Field<2>& h,
    const Field<2>& theta,
    const VectorField<2>& u0,
    const DualVectorField<2>& rhs
  ) const
  {
    VectorField<2> lambda(discretization);
    const auto& vector_dsc = lambda.get_field_discretization();

    DualVectorField<2> f(rhs);

    Vector<double>& Lambda = lambda.get_coefficients();
    Vector<double>& F = f.get_coefficients();

    auto A = velocity_matrix<linearized>(h, theta, u0, *this);
    MatrixTools::apply_boundary_values(
      vector_dsc.zero_boundary_values(), A, Lambda, F, false
    );

    linear_solve(A, Lambda, F, lambda.get_constraints());

    return lambda;
  }


} // End of icepack namespace
