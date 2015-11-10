
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <icepack/glacier_models/shallow_stream.hpp>


namespace icepack
{
  using dealii::QGauss;
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
    const auto& h_dof_handler = scalar_pde_skeleton.get_dof_handler();

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
