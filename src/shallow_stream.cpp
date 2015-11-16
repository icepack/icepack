
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/numerics/matrix_tools.h>

#include <icepack/glacier_models/shallow_stream.hpp>


namespace icepack
{
  using dealii::SymmetricTensor;
  using dealii::unit_symmetric_tensor;
  using dealii::identity_tensor;
  using dealii::outer_product;

  using dealii::FullMatrix;
  using dealii::SparseMatrix;

  using dealii::SolverControl;
  using dealii::SolverCG;
  using dealii::SparseILU;

  using dealii::FEValues;
  using dealii::FEFaceValues;
  using dealii::GeometryInfo;
  namespace FEValuesExtractors = dealii::FEValuesExtractors;

  /**
   * Constructors & destructors
   */

  ShallowStream::ShallowStream(const Triangulation<2>& tria,
                               const unsigned int p)
    :
    triangulation(tria),
    scalar_pde_skeleton(tria, FE_Q<2>(p)),
    vector_pde_skeleton(tria, FESystem<2>(FE_Q<2>(p), 2))
  {}


  /**
   * Interpolating observational data to finite element representation
   */

  Field<2>
  ShallowStream::interpolate(const Function<2>& phi) const
  {
    return icepack::interpolate(
      triangulation,
      scalar_pde_skeleton.get_fe(),
      scalar_pde_skeleton.get_dof_handler(),
      phi
    );
  }

  VectorField<2>
  ShallowStream::interpolate(const TensorFunction<1, 2>& f) const
  {
    return icepack::interpolate(
      triangulation,
      vector_pde_skeleton.get_fe(),
      vector_pde_skeleton.get_dof_handler(),
      f
    );
  }



  /*
   * Diagnostic/prognostic model solves
   */

  VectorField<2>
  ShallowStream::driving_stress(const Field<2>& s, const Field<2>& h) const
  {
    // Initialize the VectorField for the driving stress
    const auto& tau_fe = vector_pde_skeleton.get_fe();
    const auto& tau_dof_handler = vector_pde_skeleton.get_dof_handler();
    VectorField<2> tau(triangulation, tau_fe, tau_dof_handler);

    // Get the finite element & DoF handler for scalar fields
    const auto& h_fe = scalar_pde_skeleton.get_fe();

    // Find the polynomial degree of the finite element expansion and make
    // quadrature rules for cells and faces with sufficient accuracy
    const unsigned int p = tau_fe.tensor_degree();
    const QGauss<2> quad(p);
    const QGauss<1> f_quad(p);

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
    auto h_cell = scalar_pde_skeleton.get_dof_handler().begin_active();
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
            // Observe the d<0 -- this is a logical, which is technically also
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
      vector_pde_skeleton.get_constraints().distribute_local_to_global(
        cell_rhs, local_dof_indices, tau.get_coefficients()
      );
    }

    return std::move(tau);
  }


  // Forward declarations for some helper functions
  void velocity_matrix(
    SparseMatrix<double>& A,
    const ScalarPDESkeleton<2>& scalar_pde_skeleton,
    const VectorPDESkeleton<2>& vector_pde_skeleton,
    const Field<2>& s,
    const Field<2>& h,
    const Field<2>& beta,
    const VectorField<2>& u0
  );

  void linear_solve(
    const SparseMatrix<double>& A,
    Vector<double>& u,
    const Vector<double>& f,
    const ConstraintMatrix& constraints
  );

  VectorField<2> ShallowStream::diagnostic_solve(
    const Field<2>& s,
    const Field<2>& h,
    const Field<2>& beta,
    const VectorField<2>& u0
  ) const
  {
    SparseMatrix<double> A(vector_pde_skeleton.get_sparsity_pattern());

    VectorField<2> u_old = u0;
    VectorField<2> u = u0;
    auto boundary_values = vector_pde_skeleton.interpolate_boundary_values(u0);

    VectorField<2> tau = driving_stress(s, h);
    Vector<double>& F = tau.get_coefficients();
    Vector<double>& U = u.get_coefficients();
    Vector<double>& U_old = u_old.get_coefficients();

    // TODO: make these function parameters
    const double tolerance = 1.0e-2;
    const unsigned int max_iteration_count = 12;
    unsigned int iteration_count = 0;
    double error = 1.0e16;

    for (; iteration_count < max_iteration_count && error > tolerance;
         ++iteration_count) {
      // Fill the system matrix
      velocity_matrix(A, scalar_pde_skeleton, vector_pde_skeleton, s, h, beta, u);
      dealii::MatrixTools::apply_boundary_values(boundary_values, A, U, F, false);

      // Solve the linear system with the updated matrix
      linear_solve(A, U, F, vector_pde_skeleton.get_constraints());

      // Compute the relative difference between the new and old solutions
      error = dist(u, u_old) / norm(u_old);

      U_old = U;
    }

    return std::move(u);
  }


  Field<2> ShallowStream::prognostic_solve(
    const double dt,
    const Field<2>& h,
    const Field<2>& a,
    const VectorField<2>& u
  ) const
  {
    /* TODO: write this */
    return h;
  }


  VectorField<2> ShallowStream::adjoint_solve(
    const Field<2>& h,
    const Field<2>& beta,
    const Field<2>& u0,
    const VectorField<2>& f
  ) const
  {
    /* TODO: write this */
    return f;
  }


  /**
   * Accessors
   */

  const Triangulation<2>& ShallowStream::get_triangulation() const
  {
    return triangulation;
  }

  const ScalarPDESkeleton<2>& ShallowStream::get_scalar_pde_skeleton() const
  {
    return scalar_pde_skeleton;
  }

  const VectorPDESkeleton<2>& ShallowStream::get_vector_pde_skeleton() const
  {
    return vector_pde_skeleton;
  }



  /**
   * Helper functions
   */
  SymmetricTensor<4, 2> constitutive_tensor(
    const double temperature,
    const double h,
    const SymmetricTensor<2, 2> eps
  )
  {
    const SymmetricTensor<2, 2> I = unit_symmetric_tensor<2>();
    const SymmetricTensor<4, 2> II = identity_tensor<2>();
    const SymmetricTensor<4, 2> C = II + outer_product(I, I);

    const double tr = first_invariant(eps);
    const double eps_e = sqrt((eps * eps + tr * tr)/2);
    const double nu = h * viscosity(temperature, eps_e);
    return nu * C;
  }


  void velocity_matrix (
    SparseMatrix<double>& A,
    const ScalarPDESkeleton<2>& scalar_pde_skeleton,
    const VectorPDESkeleton<2>& vector_pde_skeleton,
    const Field<2>& s,
    const Field<2>& h,
    const Field<2>& beta,
    const VectorField<2>& u0
  )
  {
    A = 0;

    const auto& u_fe = vector_pde_skeleton.get_fe();
    const auto& u_dof_handler = vector_pde_skeleton.get_dof_handler();

    const auto& h_fe = scalar_pde_skeleton.get_fe();

    const unsigned int p = u_fe.tensor_degree();
    const QGauss<2> quad(p);
    const QGauss<1> f_quad(p);

    FEValues<2> u_fe_values(u_fe, quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Vector exv(0);

    FEValues<2> h_fe_values(h_fe, quad, DefaultUpdateFlags::flags);
    const FEValuesExtractors::Scalar exs(0);

    const unsigned int n_q_points = quad.size();
    const unsigned int dofs_per_cell = u_fe.dofs_per_cell;

    std::vector<double> h_values(n_q_points);
    std::vector<double> s_values(n_q_points);
    std::vector<SymmetricTensor<2, 2>> strain_rate_values(n_q_points);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    auto cell = u_dof_handler.begin_active();
    auto h_cell = scalar_pde_skeleton.get_dof_handler().begin_active();
    for (; cell != u_dof_handler.end(); ++cell, ++h_cell) {
      cell_matrix = 0;
      u_fe_values.reinit(cell);
      h_fe_values.reinit(h_cell);

      h_fe_values[exs].get_function_values(h.get_coefficients(), h_values);
      h_fe_values[exs].get_function_values(s.get_coefficients(), s_values);

      u_fe_values[exv].get_function_symmetric_gradients(
        u0.get_coefficients(), strain_rate_values
      );

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = u_fe_values.JxW(q);
        const double h = h_values[q];
        const SymmetricTensor<2, 2> eps = strain_rate_values[q];

        // TODO: use an actual temperature field
        const double T = 263.15;

        const SymmetricTensor<4, 2> Cq = constitutive_tensor(T, h, eps);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          auto eps_phi_i = u_fe_values[exv].symmetric_gradient(i, q);

          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            auto eps_phi_j = u_fe_values[exv].symmetric_gradient(j, q);

            cell_matrix(i, j) += (eps_phi_i * Cq * eps_phi_j) * dx;
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      vector_pde_skeleton.get_constraints().distribute_local_to_global(
        cell_matrix, local_dof_indices, A
      );
    }

    A.compress(dealii::VectorOperation::add);
  }


  void linear_solve(
    const SparseMatrix<double>& A,
    Vector<double>& u,
    const Vector<double>& f,
    const ConstraintMatrix& constraints
  )
  {
    SolverControl solver_control(1000, 1.0e-12);
    SolverCG<>    cg(solver_control);

    SparseILU<double> M;
    M.initialize(A);

    cg.solve(A, u, f, M);

    constraints.distribute(u);
  }

}
