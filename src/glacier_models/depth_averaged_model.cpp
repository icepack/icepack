
#include <icepack/glacier_models/depth_averaged_model.hpp>

namespace icepack {

  DepthAveragedModel::DepthAveragedModel(
    const Triangulation<2>& tria,
    const unsigned int p
  )
    :
    triangulation(tria),
    scalar_pde(tria, FE_Q<2>(p)),
    vector_pde(tria, FESystem<2>(FE_Q<2>(p), 2))
  {}


  /*
   * Interpolating observational data to finite element representation
   */

  Field<2>
  DepthAveragedModel::interpolate(const Function<2>& phi) const
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
