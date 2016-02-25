
#include <deal.II/numerics/matrix_tools.h>

#include <icepack/util/tensor_function_utils.hpp>
#include <icepack/glacier_models/depth_averaged_model.hpp>
#include <icepack/numerics/linear_solve.hpp>

namespace icepack {

  using dealii::FullMatrix;
  using dealii::FEValues;
  using dealii::FEFaceValues;
  using dealii::GeometryInfo;
  namespace FEValuesExtractors = dealii::FEValuesExtractors;

  namespace DefaultPhysicalParams {
    /**
     * Glen's flow law exponent
     */
    const double n = 3.0;
  }

  DepthAveragedModel::DepthAveragedModel(
    const Triangulation<2>& tria,
    const unsigned int p
  )
    :
    constitutive_tensor(DefaultPhysicalParams::n),
    triangulation(tria),
    scalar_pde(tria, FE_Q<2>(p)),
    vector_pde(tria, FESystem<2>(FE_Q<2>(p), 2))
  {}


  /*
   * Interpolating observational data to finite element representation
   */

  Field<2> DepthAveragedModel::interpolate(const Function<2>& phi) const
  {
    return icepack::interpolate(
      triangulation,
      scalar_pde.get_fe(),
      scalar_pde.get_dof_handler(),
      phi
    );
  }

  VectorField<2>
  DepthAveragedModel::interpolate(const TensorFunction<1, 2>& f) const
  {
    return icepack::interpolate(
      triangulation,
      vector_pde.get_fe(),
      vector_pde.get_dof_handler(),
      f
    );
  }


  VectorField<2> DepthAveragedModel::interpolate(
    const Function<2>& phi0,
    const Function<2>& phi1
  ) const
  {
    const auto phi = util::TensorFunctionFromScalarFunctions<2>(phi0, phi1);
    return interpolate(phi);
  }


  /**
   * Helper function for prognostic solve
   */
  void advection_matrix(
    SparseMatrix<double>& A,
    const double dt,
    const VectorField<2>& u,
    const DepthAveragedModel& glacier_model
  )
  {
    A = 0;

    const auto& vector_pde = glacier_model.get_vector_pde_skeleton();
    const auto& scalar_pde = glacier_model.get_scalar_pde_skeleton();

    const auto& h_fe = scalar_pde.get_fe();
    const auto& h_dof_handler = scalar_pde.get_dof_handler();

    const auto& u_fe = vector_pde.get_fe();

    const QGauss<2>& quad = scalar_pde.get_quadrature();
    const QGauss<1>& f_quad = scalar_pde.get_face_quadrature();

    FEValues<2> h_fe_values(h_fe, quad, DefaultUpdateFlags::flags);
    FEFaceValues<2> h_fe_face_values(h_fe, f_quad, DefaultUpdateFlags::face_flags);
    const FEValuesExtractors::Scalar exs(0);

    FEValues<2> u_fe_values(u_fe, quad, DefaultUpdateFlags::flags);
    FEFaceValues<2> u_fe_face_values(u_fe, f_quad, DefaultUpdateFlags::face_flags);
    const FEValuesExtractors::Vector exv(0);

    const unsigned int n_q_points = quad.size();
    const unsigned int n_face_q_points = f_quad.size();
    const unsigned int dofs_per_cell = h_fe.dofs_per_cell;

    std::vector<Tensor<1, 2> > u_values(n_q_points);
    std::vector<Tensor<1, 2> > u_face_values(n_face_q_points);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    auto cell = h_dof_handler.begin_active();
    auto u_cell = vector_pde.get_dof_handler().begin_active();
    for (; cell != h_dof_handler.end(); ++cell, ++u_cell) {
      cell_matrix = 0;
      h_fe_values.reinit(cell);
      u_fe_values.reinit(u_cell);

      u_fe_values[exv].get_function_values(u.get_coefficients(), u_values);

      // Add up contributions from the current cell
      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = h_fe_values.JxW(q);
        const Tensor<1, 2> U = u_values[q];

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const double phi_i = h_fe_values[exs].value(i, q);
          const Tensor<1, 2> d_phi_i = h_fe_values[exs].gradient(i, q);

          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const double phi_j = h_fe_values[exs].value(j, q);

            cell_matrix(i, j) +=
              (phi_i * phi_j - dt * (d_phi_i * U) * phi_j) * dx;
          }
        }
      }

      // Add up contributions from faces on the outflow boundary
      for (unsigned int face_number = 0;
           face_number < GeometryInfo<2>::faces_per_cell; ++face_number)
        if (cell->face(face_number)->at_boundary()
            and
            cell->face(face_number)->boundary_id() == 1) {
          h_fe_face_values.reinit(cell, face_number);
          u_fe_face_values.reinit(u_cell, face_number);

          u_fe_face_values[exv].get_function_values(u.get_coefficients(), u_face_values);

          for (unsigned int q = 0; q < n_face_q_points; ++q) {
            const double dl = h_fe_face_values.JxW(q);
            const Tensor<1, 2> U = u_face_values[q];
            const Tensor<1, 2> n = h_fe_face_values.normal_vector(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const double phi_i = h_fe_face_values[exs].value(i, q);

              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const double phi_j = h_fe_face_values[exs].value(j, q);

                cell_matrix(i, j) += dt * (phi_i * (U * n) * phi_j) * dl;
              }
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


  Field<2> DepthAveragedModel::prognostic_solve(
    const double dt,
    const Field<2>& h0,
    const Field<2>& a,
    const VectorField<2>& u
  ) const
  {
    Field<2> h;
    h.copy_from(h0);
    Vector<double>& H = h.get_coefficients();

    Field<2> rhs;
    rhs.copy_from(h0);
    Vector<double>& R = rhs.get_coefficients();
    R.add(dt, a.get_coefficients());

    // TODO store the mass matrix somewhere
    SparseMatrix<double> B(scalar_pde.get_sparsity_pattern());
    dealii::MatrixCreator::create_mass_matrix(
      scalar_pde.get_dof_handler(), scalar_pde.get_quadrature(), B
    );
    Vector<double> F(R.size());
    B.vmult(F, R);

    auto boundary_values = scalar_pde.interpolate_boundary_values(h0);

    SparseMatrix<double> A(scalar_pde.get_sparsity_pattern());
    advection_matrix(A, dt, u, *this);
    dealii::MatrixTools::apply_boundary_values(boundary_values, A, H, F, false);

    linear_solve<SolverBicgstab<> >(A, H, F, scalar_pde.get_constraints());

    return h;
  }


  /*
   * Accessors
   */

  const Triangulation<2>& DepthAveragedModel::get_triangulation() const
  {
    return triangulation;
  }

  const ScalarPDESkeleton<2>&
  DepthAveragedModel::get_scalar_pde_skeleton() const
  {
    return scalar_pde;
  }

  const VectorPDESkeleton<2>&
  DepthAveragedModel::get_vector_pde_skeleton() const
  {
    return vector_pde;
  }


}
