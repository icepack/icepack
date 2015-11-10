
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <icepack/glacier_models/shallow_stream.hpp>


namespace icepack
{
  using dealii::QGauss;
  using dealii::FEValues;
  using dealii::FEFaceValues;
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
    const auto& tau_fe = vector_pde_skeleton.get_fe();
    const auto& tau_dof_handler = vector_pde_skeleton.get_dof_handler();
    VectorField<2> tau(triangulation, tau_fe, tau_dof_handler);

    const auto& h_fe = scalar_pde_skeleton.get_fe();
    const auto& h_dof_handler = scalar_pde_skeleton.get_dof_handler();

    const QGauss<2> quad(2);
    const QGauss<1> f_quad(2);

    FEValues<2> tau_fe_values(tau_fe, quad, DefaultUpdateFlags::flags);
    FEFaceValues<2> tau_fe_face_values(tau_fe, f_quad, DefaultUpdateFlags::face_flags);
    const FEValuesExtractors::Vector exv(0);

    FEValues<2> h_fe_values(h_fe, quad, DefaultUpdateFlags::flags);
    FEFaceValues<2> h_fe_face_values(h_fe, f_quad, DefaultUpdateFlags::face_flags);
    const FEValuesExtractors::Scalar exs(0);

    const unsigned int n_q_points = quad.size();
    const unsigned int n_face_q_points = f_quad.size();
    const unsigned int dofs_per_cell = tau_fe.dofs_per_cell;

    std::vector<double> h_values(n_q_points);
    std::vector<Tensor<1, 2> > grad_s_values(n_q_points);

    std::vector<double> h_face_values(n_face_q_points);
    std::vector<double> s_face_values(n_face_q_points);

    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    auto cell = tau_dof_handler.begin_active();
    auto h_cell = scalar_pde_skeleton.get_dof_handler().begin_active();
    for (; cell != tau_dof_handler.end(); ++cell, ++h_cell) {
      cell_rhs = 0;
      tau_fe_values.reinit(cell);
      h_fe_values.reinit(h_cell);

      h_fe_values[exs].get_function_values(h.get_coefficients(), h_values);
      h_fe_values[exs].get_function_gradients(s.get_coefficients(), grad_s_values);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = tau_fe_values.JxW(q);
        const Tensor<1, 2> tau_q =
          -rho_ice * gravity * h_values[q] * grad_s_values[q];

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          cell_rhs(i) += tau_fe_values[exv].value(i, q) * tau_q * dx;
      }

      // TODO: add face contribution

      cell->get_dof_indices(local_dof_indices);
      vector_pde_skeleton.get_constraints().distribute_local_to_global(
        cell_rhs, local_dof_indices, tau.get_coefficients()
      );
    }

    return std::move(tau);
  }


  VectorField<2>
  ShallowStream::diagnostic_solve(
    const Field<2>& s,
    const Field<2>& h,
    const Field<2>& beta,
    const VectorField<2>& u0
  ) const
  {
    /* TODO: write this */
    return u0;
  }


  Field<2>
  ShallowStream::prognostic_solve(
    const double dt,
    const Field<2>& h,
    const Field<2>& a,
    const VectorField<2>& u
  ) const
  {
    /* TODO: write this */
    return h;
  }


  VectorField<2>
  ShallowStream::adjoint_solve(
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

}
