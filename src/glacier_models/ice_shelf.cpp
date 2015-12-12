
#include <icepack/glacier_models/ice_shelf.hpp>

namespace icepack {

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
    const auto& tau_fe = vector_pde.get_fe();
    const auto& tau_dof_handler = vector_pde.get_dof_handler();
    VectorField<2> tau(triangulation, tau_fe, tau_dof_handler);

    // TODO: write this

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


  Field<2> IceShelf::prognostic_solve(
    const double dt,
    const Field<2>& h0,
    const Field<2>& a,
    const VectorField<2>& u
  ) const
  {
    Field<2> h;
    h.copy_from(h0);

    // TODO: write this

    return h;
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
