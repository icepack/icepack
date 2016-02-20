
#include <deal.II/base/exceptions.h>

#include <icepack/util/tensor_function_utils.hpp>
#include <icepack/glacier_models/depth_averaged_model.hpp>

namespace icepack {

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


  Field<2> DepthAveragedModel::prognostic_solve(
    const double dt,
    const Field<2>& h0,
    const Field<2>& a,
    const VectorField<2>& u
  ) const
  {
    Field<2> h;
    h.copy_from(h0);

    /* TODO: write this. */
    // Throw an error since this isn't implemented.
    Assert(false, dealii::ExcInternalError());

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
