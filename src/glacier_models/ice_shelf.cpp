
#include <icepack/physics/constants.hpp>
#include <icepack/glacier_models/ice_shelf.hpp>

namespace icepack {

  using dealii::FullMatrix;
  using dealii::SparseMatrix;

  using dealii::FEValues;
  using dealii::FEFaceValues;
  namespace FEValuesExtractors = dealii::FEValuesExtractors;


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
      h_fe_values.reinit(cell);

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

    // TODO: write this

    return r;
  }


  VectorField<2> IceShelf::diagnostic_solve(
    const Field<2>& h,
    const VectorField<2>& u0
  ) const
  {
    VectorField<2> u;
    u.copy_from(u0);

    // TODO: write this

    return u;
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
