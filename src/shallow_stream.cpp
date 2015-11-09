
#include <icepack/glacier_models/shallow_stream.hpp>


namespace icepack
{
  /**
   * Constructors & destructors
   */

  ShallowStream::ShallowStream(const Triangulation<2>& tri, const unsigned int p)
    :
    triangulation(tri),
    scalar_finite_element(p),
    vector_finite_element(scalar_finite_element, 2),
    scalar_dof_handler(triangulation),
    vector_dof_handler(triangulation)
  {
    scalar_dof_handler.distribute_dofs(scalar_finite_element);
    vector_dof_handler.distribute_dofs(vector_finite_element);
  }

  ShallowStream::~ShallowStream()
  {
    scalar_dof_handler.clear();
    vector_dof_handler.clear();
  }


  /**
   * Interpolating observational data to finite element representation
   */

  Field<2>
  ShallowStream::interpolate(const Function<2>& phi) const
  {
    return icepack::interpolate(
      triangulation,
      scalar_finite_element,
      scalar_dof_handler,
      phi
    );
  }

  VectorField<2>
  ShallowStream::interpolate(const TensorFunction<1, 2>& f) const
  {
    return icepack::interpolate(
      triangulation,
      vector_finite_element,
      vector_dof_handler,
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

  const FE_Q<2>& ShallowStream::get_scalar_fe() const
  {
    return scalar_finite_element;
  }

  const FESystem<2>& ShallowStream::get_vector_fe() const
  {
    return vector_finite_element;
  }

  const DoFHandler<2>& ShallowStream::get_scalar_dof_handler() const
  {
    return scalar_dof_handler;
  }

  const DoFHandler<2>& ShallowStream::get_vector_dof_handler() const
  {
    return vector_dof_handler;
  }

}
