
#ifndef ICEPACK_ICE_SHELF_HPP
#define ICEPACK_ICE_SHELF_HPP

#include <icepack/glacier_models/depth_averaged_model.hpp>

namespace icepack {

  /**
   * This class is for modeling the flow of floating ice shelves.
   */
  class IceShelf : public DepthAveragedModel
  {
  public:
    using DepthAveragedModel::DepthAveragedModel;

    /**
     * Compute the driving stress
     \f[
     \tau = -\rho gh\nabla s
     \f]
     * from the ice thickness \f$h\f$ and surface elevation \f$s\f$.
     */
    DualVectorField<2> driving_stress(const Field<2>& thickness) const;


    /**
     * Computed the value of the action for the shallow shelf equations. The
     * solution of the diagnostic equations is the unique extremizer of the
     * action.
     */
    double action(
      const Field<2>& thickness,
      const Field<2>& temperature,
      const VectorField<2>& velocity
    ) const;


    /**
     * Compute the residual of a candidate solution to the diagnostic equation.
     * This vector is used to solve the system by Newton's method.
     */
    DualVectorField<2> residual(
      const Field<2>& thickness,
      const Field<2>& temperature,
      const VectorField<2>& u,
      const DualVectorField<2>& tau_d
    ) const;

    /**
     * Compute the ice velocity from the thickness and temperature
     */
    VectorField<2> diagnostic_solve(
      const Field<2>& thickness,
      const Field<2>& temperature,
      const VectorField<2>& u0
    ) const;

    /*
     * Field<2> prognostic_solve(...) const
     *
     * Prognostic solves for ice shelves are no different from that for any
     * other depth-averaged glacier model, so we use the implementation defined
     * in DepthAveragedModel.
     */

    /**
     * Solve the linearization of the diagnostic equation around a velocity u0
     * with a given VectorField as right-hand side
     */
    VectorField<2> adjoint_solve(
      const Field<2>& thickness,
      const Field<2>& temperature,
      const VectorField<2>& u0,
      const DualVectorField<2>& f
    ) const;
  };

}

#endif
