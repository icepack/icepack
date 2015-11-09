
#include <icepack/glacier_models/shallow_stream.hpp>


namespace icepack
{
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
  ShallowStream::diagnostic_solve(
    const Field<2>& s,
    const Field<2>& h,
    const Field<2>& beta,
    const VectorField<2>& u0
  )
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
  )
  {
    /* TODO: write this */
    return h;
  }


  VectorField<2>
  ShallowStream::adjoint_solve(
    const Field<2>& h,
    const Field<2>& beta,
    const Field<2>& u0,
    const VectorField<2>& f)
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
