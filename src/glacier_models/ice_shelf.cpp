
#include <deal.II/numerics/matrix_tools.h>

#include <icepack/physics/constants.hpp>
#include <icepack/physics/viscosity.hpp>
#include <icepack/numerics/linear_solve.hpp>
#include <icepack/glacier_models/ice_shelf.hpp>

namespace icepack {

  using dealii::FullMatrix;

  using dealii::FEValues;
  using dealii::FEFaceValues;
  namespace FEValuesExtractors = dealii::FEValuesExtractors;


  /* ================
   * Helper functions
   * ================ */

  /**
   * Construct the system matrix for the diagnostic equations
   */
  template <class ConstitutiveTensor>
  void velocity_matrix(
    SparseMatrix<double>& A,
    const Field<2>& h,
    const VectorField<2>& u0,
    const IceShelf& ice_shelf,
    const ConstitutiveTensor constitutive_tensor
  )
  {
    A = 0;

    const auto& vector_pde = ice_shelf.get_vector_pde_skeleton();
    const auto& scalar_pde = ice_shelf.get_scalar_pde_skeleton();

    const auto& u_fe = vector_pde.get_fe();
    const auto& u_dof_handler = vector_pde.get_dof_handler();

    const auto& h_fe = scalar_pde.get_fe();

    const QGauss<2>& quad = vector_pde.get_quadrature();

    FEValues<2> u_fe_values(u_fe, quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Vector exv(0);

    FEValues<2> h_fe_values(h_fe, quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Scalar exs(0);

    const unsigned int n_q_points = quad.size();
    const unsigned int dofs_per_cell = u_fe.dofs_per_cell;

    std::vector<double> h_values(n_q_points);
    std::vector<SymmetricTensor<2, 2>> strain_rate_values(n_q_points);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    auto cell = u_dof_handler.begin_active();
    auto h_cell = scalar_pde.get_dof_handler().begin_active();
    for (; cell != u_dof_handler.end(); ++cell, ++h_cell) {
      cell_matrix = 0;
      u_fe_values.reinit(cell);
      h_fe_values.reinit(h_cell);

      h_fe_values[exs].get_function_values(h.get_coefficients(), h_values);
      u_fe_values[exv].get_function_symmetric_gradients(
        u0.get_coefficients(), strain_rate_values
      );

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = u_fe_values.JxW(q);
        const double H = h_values[q];
        const SymmetricTensor<2, 2> eps = strain_rate_values[q];

        // TODO: use an actual temperature field
        const double T = 263.13;

        const SymmetricTensor<4, 2> C = constitutive_tensor(T, H, eps);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const auto eps_phi_i = u_fe_values[exv].symmetric_gradient(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const auto eps_phi_j = u_fe_values[exv].symmetric_gradient(j, q);
            cell_matrix(i, j) += (eps_phi_i * C * eps_phi_j) * dx;
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      vector_pde.get_constraints().distribute_local_to_global(
        cell_matrix, local_dof_indices, A
      );
    }

    A.compress(dealii::VectorOperation::add);
  }


  /**
   * Solve the diagnostic equations using Picard's method.
   */
  VectorField<2> picard_solve(
    const Field<2>& h,
    const VectorField<2>& u0,
    const IceShelf& ice_shelf,
    const double tolerance,
    const unsigned int max_iterations
  )
  {
    const auto& vector_pde = ice_shelf.get_vector_pde_skeleton();
    SparseMatrix<double> A(vector_pde.get_sparsity_pattern());

    VectorField<2> u, u_old;
    u_old.copy_from(u0);  u.copy_from(u0);
    auto boundary_values = vector_pde.interpolate_boundary_values(u0);

    VectorField<2> tau = ice_shelf.driving_stress(h);
    Vector<double>& F = tau.get_coefficients();
    Vector<double>& U = u.get_coefficients();
    Vector<double>& U_old = u_old.get_coefficients();

    double error = 1.0e16;
    for (unsigned int i = 0; i < max_iterations && error > tolerance; ++i) {
      // Fill the system matrix
      velocity_matrix(A, h, u, ice_shelf, SSA::nonlinear);
      dealii::MatrixTools::apply_boundary_values(boundary_values, A, U, F, false);

      // Solve the linear system with the updated matrix
      linear_solve(A, U, F, vector_pde.get_constraints());

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
    const auto& tau_fe = vector_pde.get_fe();
    const auto& tau_dof_handler = vector_pde.get_dof_handler();
    VectorField<2> tau(triangulation, tau_fe, tau_dof_handler);

    const auto& h_fe = scalar_pde.get_fe();

    const QGauss<2>& quad = vector_pde.get_quadrature();
    const QGauss<1>& f_quad = vector_pde.get_face_quadrature();

    FEValues<2> tau_fe_values(tau_fe, quad, DefaultUpdateFlags::flags);
    FEFaceValues<2> tau_fe_face_values(tau_fe, f_quad, DefaultUpdateFlags::face_flags);
    const FEValuesExtractors::Vector exv(0);

    FEValues<2> h_fe_values(h_fe, quad, DefaultUpdateFlags::flags);
    FEFaceValues<2> h_fe_face_values(h_fe, f_quad, DefaultUpdateFlags::face_flags);
    const FEValuesExtractors::Scalar exs(0);

    // Initialize storage for cell- and face-local data
    const unsigned int n_q_points = quad.size();
    const unsigned int n_face_q_points = f_quad.size();
    const unsigned int dofs_per_cell = tau_fe.dofs_per_cell;

    std::vector<double> h_values(n_q_points);
    std::vector<double> h_face_values(n_face_q_points);

    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    const double Rho = rho_ice * gravity * (1 - rho_ice / rho_water);

    // Create cell iterators from the tau and h DoFHandlers; these will be
    // iterated jointly.
    auto cell = tau_dof_handler.begin_active();
    auto h_cell = scalar_pde.get_dof_handler().begin_active();
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

        // NOTE: check +/- signs
        const double tau_q = 0.5 * Rho * H * H;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          cell_rhs(i) += tau_q * tau_fe_values[exv].divergence(i, q) * dx;
      }

      // NOTE: do we no longer need boundary integrals?

      cell->get_dof_indices(local_dof_indices);
      vector_pde.get_constraints().distribute_local_to_global(
        cell_rhs, local_dof_indices, tau.get_coefficients()
      );
    }

    return tau;
  }


  VectorField<2> IceShelf::residual(
    const Field<2>& h,
    const VectorField<2>& u,
    const VectorField<2>& f
  ) const
  {
    VectorField<2> r;
    r.copy_from(f);

    const auto& u_fe = vector_pde.get_fe();
    const auto& u_dof_handler = vector_pde.get_dof_handler();

    const auto& h_fe = scalar_pde.get_fe();

    const QGauss<2>& quad = vector_pde.get_quadrature();

    FEValues<2> u_fe_values(u_fe, quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Vector exv(0);

    FEValues<2> h_fe_values(h_fe, quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Scalar exs(0);

    const unsigned int n_q_points = quad.size();
    const unsigned int dofs_per_cell = u_fe.dofs_per_cell;

    std::vector<double> h_values(n_q_points);
    std::vector<Tensor<1, 2>> u_values(n_q_points);
    std::vector<SymmetricTensor<2, 2>> strain_rate_values(n_q_points);

    Vector<double> cell_residual(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    auto cell = u_dof_handler.begin_active();
    auto h_cell = scalar_pde.get_dof_handler().begin_active();
    for (; cell != u_dof_handler.end(); ++cell, ++h_cell) {
      cell_residual = 0;
      u_fe_values.reinit(cell);
      h_fe_values.reinit(h_cell);

      h_fe_values[exs].get_function_values(h.get_coefficients(), h_values);
      u_fe_values[exv].get_function_symmetric_gradients(
        u.get_coefficients(), strain_rate_values
      );

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = u_fe_values.JxW(q);
        const double H = h_values[q];
        const SymmetricTensor<2, 2> eps = strain_rate_values[q];

        // TODO: use an actual temperature field
        const double T = 263.15;

        const SymmetricTensor<4, 2> C = SSA::nonlinear(T, H, eps);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const auto eps_phi_i = u_fe_values[exv].symmetric_gradient(i, q);
          cell_residual(i) -= (eps_phi_i * C * eps) * dx;
        }
      }

      cell->get_dof_indices(local_dof_indices);
      vector_pde.get_constraints().distribute_local_to_global(
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
    const VectorField<2>& u0
  ) const
  {
    return picard_solve(h, u0, *this, 0.001, 20);
  }


  VectorField<2> IceShelf::adjoint_solve(
    const Field<2>& h,
    const VectorField<2>& u0,
    const VectorField<2>& f
  ) const
  {
    VectorField<2> q;
    q.copy_from(f);

    // TODO: write this

    return q;
  }


} // End of icepack namespace
