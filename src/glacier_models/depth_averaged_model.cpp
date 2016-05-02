
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/filtered_matrix.h>

#include <icepack/util/tensor_function_utils.hpp>
#include <icepack/glacier_models/depth_averaged_model.hpp>
#include <icepack/numerics/linear_solve.hpp>

namespace icepack {

  using dealii::FullMatrix;
  using dealii::FEValues;
  using dealii::FEFaceValues;
  using dealii::GeometryInfo;
  using dealii::FilteredMatrix;
  namespace FEValuesExtractors = dealii::FEValuesExtractors;

  using DefaultUpdateFlags::flags;
  using DefaultUpdateFlags::face_flags;

  namespace DefaultPhysicalParams {
    /**
     * Glen's flow law exponent
     */
    const double n = 3.0;
  }

  DepthAveragedModel::DepthAveragedModel(
    const Triangulation<2>& tria,
    const unsigned int p
  ) :
    constitutive_tensor(DefaultPhysicalParams::n),
    discretization(tria, p)
  {}


  /*
   * Interpolating observational data to finite element representation
   */

  Field<2> DepthAveragedModel::interpolate(const Function<2>& phi) const
  {
    return icepack::interpolate(discretization, phi);
  }


  VectorField<2>
  DepthAveragedModel::interpolate(const TensorFunction<1, 2>& f) const
  {
    return icepack::interpolate(discretization, f);
  }


  VectorField<2> DepthAveragedModel::interpolate(
    const Function<2>& phi0,
    const Function<2>& phi1
  ) const
  {
    const auto phi =
      internal::TensorFunctionFromScalarFunctions<2>(phi0, phi1);
    return interpolate(phi);
  }


  /**
   * Helper function for prognostic solve
   */
  Field<2> DepthAveragedModel::dh_dt(
    const Field<2>& h0,
    const Field<2>& a,
    const VectorField<2>& u
  ) const
  {
    Field<2> dh(discretization);

    const auto& h_fe = dh.get_fe();
    const auto& u_fe = u.get_fe();

    const auto& h_dof_handler = dh.get_dof_handler();

    const QGauss<2> quad = discretization.quad();
    const QGauss<1> f_quad = discretization.face_quad();

    FEValues<2> h_fe_values(h_fe, quad, flags);
    FEFaceValues<2> h_fe_face_values(h_fe, f_quad, face_flags);
    const FEValuesExtractors::Scalar exs(0);

    FEValues<2> u_fe_values(u_fe, quad, flags);
    FEFaceValues<2> u_fe_face_values(u_fe, f_quad, face_flags);
    const FEValuesExtractors::Vector exv(0);

    const unsigned int n_q_points = quad.size();
    const unsigned int n_face_q_points = f_quad.size();
    const unsigned int dofs_per_cell = h_fe.dofs_per_cell;

    std::vector<double> a_values(n_q_points);
    std::vector<double> h_values(n_q_points);
    std::vector<Tensor<1, 2> > dh_values(n_q_points);
    std::vector<double> h_face_values(n_face_q_points);
    std::vector<Tensor<1, 2> > dh_face_values(n_face_q_points);
    std::vector<Tensor<1, 2> > u_values(n_q_points);
    std::vector<Tensor<1, 2> > u_face_values(n_face_q_points);

    // Compute a natural time scale for this grid and velocity
    const double u_max = u.get_coefficients().linfty_norm();
    const double dx_min =
      dealii::GridTools::minimal_cell_diameter(get_triangulation());

    // TODO check the factor of 4.0. This should probably be the minimal length
    // of one of the triangulation edges but deal.II only gives us the cell
    // diameter, which is an overestimate by sqrt(2) even for a square cell.
    const double tau = dx_min / u_max / 4.0;

    Vector<double> cell_dh(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    auto cell = h_dof_handler.begin_active();
    auto u_cell = u.get_dof_handler().begin_active();
    for (; cell != h_dof_handler.end(); ++cell, ++u_cell) {
      cell_dh = 0;
      h_fe_values.reinit(cell);
      u_fe_values.reinit(u_cell);

      h_fe_values[exs].get_function_values(h0.get_coefficients(), h_values);
      h_fe_values[exs].get_function_gradients(h0.get_coefficients(), dh_values);
      h_fe_values[exs].get_function_values(a.get_coefficients(), a_values);
      u_fe_values[exv].get_function_values(u.get_coefficients(), u_values);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = h_fe_values.JxW(q);
        const double A = a_values[q];
        const double H = h_values[q];
        const Tensor<1, 2> dH = dh_values[q];
        const Tensor<1, 2> U = u_values[q];

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const double phi_i = h_fe_values[exs].value(i, q);
          const Tensor<1, 2> d_phi_i = h_fe_values[exs].gradient(i, q);

          const Tensor<1, 2> a_flux = H * U;
          const Tensor<1, 2> d_flux = -tau * (U * dH) * U;

          cell_dh(i) += (phi_i * A + d_phi_i * (a_flux + d_flux)) * dx;
        }
      }

      for (unsigned int face_number = 0;
           face_number < GeometryInfo<2>::faces_per_cell; ++face_number)
        if (cell->face(face_number)->at_boundary()) {
          h_fe_face_values.reinit(cell, face_number);
          u_fe_face_values.reinit(u_cell, face_number);

          h_fe_face_values[exs].get_function_values(h0.get_coefficients(), h_face_values);
          u_fe_face_values[exv].get_function_values(u.get_coefficients(), u_face_values);

          for (unsigned int q = 0; q < n_face_q_points; ++q) {
            const double dl = h_fe_face_values.JxW(q);
            const double H = h_face_values[q];
            const Tensor<1, 2> dH = dh_face_values[q];
            const Tensor<1, 2> U = u_face_values[q];
            const Tensor<1, 2> n = h_fe_face_values.normal_vector(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const double phi_i = h_fe_face_values[exs].value(i, q);

              const Tensor<1, 2> a_flux = H * U;
              const Tensor<1, 2> d_flux = -tau * (U * dH) * U;

              cell_dh(i) -= phi_i * (a_flux + d_flux) * n * dl;
            }
          }
        }

      cell->get_dof_indices(local_dof_indices);
      dh.get_constraints().distribute_local_to_global(
        cell_dh, local_dof_indices, dh.get_coefficients()
      );
    }

    Vector<double> F(dh.get_coefficients());
    SolverControl solver_control(1000, 1.0e-10);
    solver_control.log_result(false);
    SolverCG<> solver(solver_control);

    const auto& M = discretization.scalar().get_mass_matrix();
    solver.solve(M, dh.get_coefficients(), F, dealii::PreconditionIdentity());
    dh.get_constraints().distribute(dh.get_coefficients());

    return dh;
  }


  Field<2> DepthAveragedModel::prognostic_solve(
    const double dt,
    const Field<2>& h0,
    const Field<2>& a,
    const VectorField<2>& u
  ) const
  {
    Field<2> h_dot = dh_dt(h0, a, u);
    const auto& boundary_values =
      h0.get_field_discretization().zero_boundary_values();

    for (const auto& p: boundary_values)
      h_dot.get_coefficients()[p.first] = p.second;

    return h0 + dt * h_dot;
  }


  /*
   * Accessors
   */

  const Discretization<2>& DepthAveragedModel::get_discretization() const
  {
    return discretization;
  }

  const Triangulation<2>& DepthAveragedModel::get_triangulation() const
  {
    return discretization.get_triangulation();
  }

} // End of namespace icepack
