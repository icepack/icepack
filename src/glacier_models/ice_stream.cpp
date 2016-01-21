
#include <deal.II/numerics/matrix_tools.h>

#include <icepack/physics/constants.hpp>
#include <icepack/numerics/linear_solve.hpp>
#include <icepack/glacier_models/ice_stream.hpp>

namespace icepack {

  using dealii::FullMatrix;

  using dealii::FEValues;
  using dealii::FEFaceValues;
  using dealii::GeometryInfo;
  namespace FEValuesExtractors = dealii::FEValuesExtractors;


  /* ================
   * Helper functions
   * ================ */

  /**
   * Make some default BasalShear function objects
   */
  namespace {
    namespace DefaultPhysicalParams {
      const double m = 3.0, tau0 = 0.1, u0 = 100.0;
    }
  }


  /**
   * Construct the system matrix for the ice stream equations.
   */
  template <class ConstitutiveTensor, class BasalShear>
  void velocity_matrix (
    SparseMatrix<double>& A,
    const Field<2>& s,
    const Field<2>& h,
    const Field<2>& beta,
    const VectorField<2>& u0,
    const IceStream& ice_stream,
    const ConstitutiveTensor constitutive_tensor,
    const BasalShear basal_shear
  )
  {
    A = 0;

    const auto& vector_pde = ice_stream.get_vector_pde_skeleton();
    const auto& scalar_pde = ice_stream.get_scalar_pde_skeleton();

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
    std::vector<double> s_values(n_q_points);
    std::vector<double> beta_values(n_q_points);
    std::vector<Tensor<1, 2>> u_values(n_q_points);
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
      h_fe_values[exs].get_function_values(s.get_coefficients(), s_values);
      h_fe_values[exs].get_function_values(beta.get_coefficients(), beta_values);

      u_fe_values[exv].get_function_values(u0.get_coefficients(), u_values);
      u_fe_values[exv].get_function_symmetric_gradients(
        u0.get_coefficients(), strain_rate_values
      );

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = u_fe_values.JxW(q);
        const double H = h_values[q];
        const SymmetricTensor<2, 2> eps = strain_rate_values[q];

        // TODO: use an actual temperature field
        const double T = 263.15;

        const SymmetricTensor<4, 2> C = constitutive_tensor(T, H, eps);

        const double flotation = (1 - rho_ice/rho_water) * H;
        const double flotation_tolerance = 1.0e-4;
        const bool floating = s_values[q] / flotation - 1.0 > flotation_tolerance;
        const auto K = floating * basal_shear(beta_values[q], u_values[q]);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const auto eps_i = u_fe_values[exv].symmetric_gradient(i, q);
          const auto phi_i = u_fe_values[exv].value(i, q);

          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const auto eps_j = u_fe_values[exv].symmetric_gradient(j, q);
            const auto phi_j = u_fe_values[exv].value(j, q);

            cell_matrix(i, j) += (eps_i * C * eps_j + phi_i * (K * phi_j)) * dx;
          }
        }
      }

      // Add the local stiffness matrix to the global stiffness matrix
      cell->get_dof_indices(local_dof_indices);
      vector_pde.get_constraints().distribute_local_to_global(
        cell_matrix, local_dof_indices, A
      );
    }

    A.compress(dealii::VectorOperation::add);
  }


  /**
   * Solve the diagnostic equations using Newton's method.
   */
  VectorField<2> newton_solve(
    const Field<2>& s,
    const Field<2>& h,
    const Field<2>& beta,
    const VectorField<2>& u0,
    const IceStream& ice_stream,
    const double tolerance,
    const unsigned int max_iterations
  )
  {
    const auto& vector_pde = ice_stream.get_vector_pde_skeleton();
    SparseMatrix<double> A(vector_pde.get_sparsity_pattern());

    VectorField<2> u;
    u.copy_from(u0);
    auto boundary_values = vector_pde.zero_boundary_values();

    const VectorField<2> tau = ice_stream.driving_stress(s, h);
    const double tau_norm = norm(tau);

    VectorField<2> r = ice_stream.residual(s, h, beta, u, tau);
    Vector<double>& R = r.get_coefficients();

    Vector<double>& U = u.get_coefficients();
    Vector<double> dU(vector_pde.get_dof_handler().n_dofs());

    const auto constitutive_tensor =
      [&](const double T, const double h, const SymmetricTensor<2, 2>& eps)
      { return ice_stream.constitutive_tensor.linearized(T, h, eps); };

    const auto basal_shear =
      [&](const double beta, const Tensor<1, 2>& u)
      { return ice_stream.basal_shear.linearized(beta, u); };

    double error = 1.0e16;
    for (unsigned int i = 0; i < max_iterations && error > tolerance; ++i) {
      // Fill the system matrix
      velocity_matrix(
        A, s, h, beta, u, ice_stream, constitutive_tensor, basal_shear
      );
      dealii::MatrixTools::apply_boundary_values(boundary_values, A, dU, R, false);

      // Solve the linear system with the updated matrix
      linear_solve(A, dU, R, vector_pde.get_constraints());
      U.add(1.0, dU);

      // Compute the relative difference between the new and old solutions
      r = ice_stream.residual(s, h, beta, u, tau);
      error = norm(r) / tau_norm;
    }

    return u;
  }


  /**
   * Solve the diagnostic equations using Picard's method.
   * The error tolerance specified is the relative change in the solution from
   * one iteration to the next.
   */
  VectorField<2> picard_solve(
    const Field<2>& s,
    const Field<2>& h,
    const Field<2>& beta,
    const VectorField<2>& u0,
    const IceStream& ice_stream,
    const double tolerance,
    const unsigned int max_iterations
  )
  {
    const auto& vector_pde = ice_stream.get_vector_pde_skeleton();
    SparseMatrix<double> A(vector_pde.get_sparsity_pattern());

    VectorField<2> u, u_old;
    u_old.copy_from(u0);  u.copy_from(u0);
    auto boundary_values = vector_pde.interpolate_boundary_values(u0);

    VectorField<2> tau = ice_stream.driving_stress(s, h);
    Vector<double>& F = tau.get_coefficients();
    Vector<double>& U = u.get_coefficients();
    Vector<double>& U_old = u_old.get_coefficients();

    const auto constitutive_tensor =
      [&](const double T, const double h, const SymmetricTensor<2, 2>& eps)
      { return ice_stream.constitutive_tensor.nonlinear(T, h, eps); };

    const auto basal_shear =
      [&](const double beta, const Tensor<1, 2>& u)
      { return ice_stream.basal_shear.nonlinear(beta, u); };

    double error = 1.0e16;
    for (unsigned int i = 0; i < max_iterations && error > tolerance; ++i) {
      // Fill the system matrix
      velocity_matrix(
        A, s, h, beta, u, ice_stream, constitutive_tensor, basal_shear
      );
      dealii::MatrixTools::apply_boundary_values(boundary_values, A, U, F, false);

      // Solve the linear system with the updated matrix
      linear_solve(A, U, F, vector_pde.get_constraints());

      // Compute the relative change in the solution
      error = dist(u, u_old) / norm(u);
      U_old = U;
    }

    return u;
  }



  /* =============================
   * Member functions of IceStream
   * ============================= */


  IceStream::IceStream(const Triangulation<2>& tria, const unsigned int p)
    :
    DepthAveragedModel(tria, p),
    basal_shear(
      DefaultPhysicalParams::m,
      DefaultPhysicalParams::u0,
      DefaultPhysicalParams::tau0
    )
  {}



  /*
   * Diagnostic/prognostic model solves
   */

  VectorField<2>
  IceStream::driving_stress(const Field<2>& s, const Field<2>& h) const
  {
    // Initialize the VectorField for the driving stress
    const auto& tau_fe = vector_pde.get_fe();
    const auto& tau_dof_handler = vector_pde.get_dof_handler();
    VectorField<2> tau(triangulation, tau_fe, tau_dof_handler);

    // Get the finite element & DoF handler for scalar fields
    const auto& h_fe = scalar_pde.get_fe();

    // Find the polynomial degree of the finite element expansion and make
    // quadrature rules for cells and faces with sufficient accuracy
    const QGauss<2>& quad = vector_pde.get_quadrature();
    const QGauss<1>& f_quad = vector_pde.get_face_quadrature();

    // Get FEValues objects and an extractor for the driving stress and the
    // thickness/surface elevation fields.
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
    std::vector<Tensor<1, 2> > grad_s_values(n_q_points);

    std::vector<double> h_face_values(n_face_q_points);
    std::vector<double> s_face_values(n_face_q_points);

    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Create cell iterators from the tau and h DoFHandlers; these will be
    // iterated jointly.
    auto cell = tau_dof_handler.begin_active();
    auto h_cell = scalar_pde.get_dof_handler().begin_active();
    for (; cell != tau_dof_handler.end(); ++cell, ++h_cell) {
      cell_rhs = 0;
      tau_fe_values.reinit(cell);
      h_fe_values.reinit(h_cell);

      h_fe_values[exs].get_function_values(h.get_coefficients(), h_values);
      h_fe_values[exs].get_function_gradients(s.get_coefficients(), grad_s_values);

      // Add up the driving stress contributions to the cell right-hand side
      // from each quadrature point.
      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = tau_fe_values.JxW(q);
        const Tensor<1, 2> tau_q =
          -rho_ice * gravity * h_values[q] * grad_s_values[q];

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          cell_rhs(i) += tau_fe_values[exv].value(i, q) * tau_q * dx;
      }

      // If we're at the calving terminus, add in the frontal stress.
      for (unsigned int face_number = 0;
           face_number < GeometryInfo<2>::faces_per_cell; ++face_number)
        if (cell->face(face_number)->at_boundary()
            and
            cell->face(face_number)->boundary_id() == 1) {
          tau_fe_face_values.reinit(cell, face_number);
          h_fe_face_values.reinit(h_cell, face_number);

          h_fe_face_values[exs].get_function_values(h.get_coefficients(), h_face_values);
          h_fe_face_values[exs].get_function_values(s.get_coefficients(), s_face_values);

          for (unsigned int q = 0; q < n_face_q_points; ++q) {
            const double dl = tau_fe_face_values.JxW(q);
            const double H = h_face_values[q];
            const double D = s_face_values[q] - H;
            const Tensor<1, 2> n = h_fe_face_values.normal_vector(q);

            // Compute the stress at the ice terminus.
            // Observe the d<0 -- this is a boolean, which is technically also
            // an integer, which equals 1 when the ice base is below sea level
            // and 0 otherwise. There's no water pressure if it's a land-
            // terminating glacier.
            const Tensor<1, 2> tau_q =
              0.5 * gravity * (rho_ice * H * H - (D<0) * rho_water * D * D) * n;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_rhs(i) += tau_fe_face_values[exv].value(i, q) * tau_q * dl;
          }
        }

      cell->get_dof_indices(local_dof_indices);
      vector_pde.get_constraints().distribute_local_to_global(
        cell_rhs, local_dof_indices, tau.get_coefficients()
      );
    }

    return tau;
  }


  VectorField<2> IceStream::residual(
    const Field<2>& s,
    const Field<2>& h,
    const Field<2>& beta,
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
    std::vector<double> s_values(n_q_points);
    std::vector<double> beta_values(n_q_points);
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
      h_fe_values[exs].get_function_values(s.get_coefficients(), s_values);
      h_fe_values[exs].get_function_values(beta.get_coefficients(), beta_values);

      u_fe_values[exv].get_function_values(u.get_coefficients(), u_values);
      u_fe_values[exv].get_function_symmetric_gradients(
        u.get_coefficients(), strain_rate_values
      );

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = u_fe_values.JxW(q);
        const double h = h_values[q];
        const SymmetricTensor<2, 2> eps = strain_rate_values[q];
        const Tensor<1, 2> u = u_values[q];

        // TODO: use an actual temperature field
        const double T = 263.15;

        const SymmetricTensor<4, 2> C =
          constitutive_tensor.nonlinear(T, h, eps);

        const double flotation = (1 - rho_ice/rho_water) * h;
        const double flotation_tolerance = 1.0e-4;
        const bool floating = s_values[q]/flotation - 1.0 > flotation_tolerance;
        const double K =
          floating * basal_shear.nonlinear(beta_values[q], u_values[q]);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const auto eps_phi_i = u_fe_values[exv].symmetric_gradient(i, q);
          const auto phi_i = u_fe_values[exv].value(i, q);
          cell_residual(i) -= (eps_phi_i * C * eps + phi_i * K * u) * dx;
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


  VectorField<2> IceStream::diagnostic_solve(
    const Field<2>& s,
    const Field<2>& h,
    const Field<2>& beta,
    const VectorField<2>& u0
  ) const
  {
    auto u = picard_solve(s, h, beta, u0, *this, 0.1, 5);
    return newton_solve(s, h, beta, u, *this, 1.0e-10, 100);
  }


  std::pair<Field<2>, Field<2> > IceStream::prognostic_solve(
    const double dt,
    const Field<2>& b,
    const Field<2>& h0,
    const Field<2>& a,
    const VectorField<2>& u
  ) const
  {
    // Invoke the prognostic solve procedure defined by DepthAveragedModel
    // in order to update the ice thickness.
    Field<2> h = static_cast<const DepthAveragedModel&>(*this)
      .prognostic_solve(dt, h0, a, u);

    // Update the ice surface elevation using the known bed elevation and the
    // updated thickness:
    //     s(x) = min{b(x) + h(x), (1 - rho_i / rho_w) * h(x)}
    Field<2> s;
    s.copy_from(b);
    // TODO: implement this
    s.get_coefficients().add(1.0, h.get_coefficients());

    return std::make_pair(std::move(h), std::move(s));
  }


  VectorField<2> IceStream::adjoint_solve(
    const Field<2>& h,
    const Field<2>& beta,
    const VectorField<2>& u0,
    const VectorField<2>& f
  ) const
  {
    VectorField<2> q;
    q.copy_from(f);

    /* TODO: write this */

    return q;
  }


} // End of icepack namespace
