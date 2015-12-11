
#ifndef ICEPACK_ICE_STREAM_HPP
#define ICEPACK_ICE_STREAM_HPP

#include <icepack/physical_constants.hpp>
#include <icepack/glacier_models/depth_averaged_model.hpp>

namespace icepack {

  /**
   * This class solves the shallow stream model of glacier flow, appropriate
   * for ice streams and ice shelves which flow with little to no vertical
   * shear.
   */
  class IceStream : public DepthAveragedModel
  {
  public:

    /**
     * Construct a model object for a given geometry and finite element basis.
     */
    IceStream(
      const Triangulation<2>& triangulation,
      const unsigned int polynomial_order
    );


    /**
     * Compute the driving stress
     \f[
     \tau = -\rho gh\nabla s
     \f]
     * from the ice thickness \f$h\f$ and surface elevation \f$s\f$.
     */
    VectorField<2> driving_stress(
      const Field<2>& surface,
      const Field<2>& thickness
    ) const;

    /**
     * Compute the residual of a candidate solution to the diagnostic equation.
     * This vector is used to solve the system by Newton's method.
     */
    VectorField<2> residual(
      const Field<2>& surface,
      const Field<2>& thickness,
      const Field<2>& beta,
      const VectorField<2>& u,
      const VectorField<2>& tau_d
    ) const;

    /**
     * Compute the ice velocity from the thickness and friction coefficient.
     */
    VectorField<2> diagnostic_solve(
      const Field<2>& surface,
      const Field<2>& thickness,
      const Field<2>& beta,
      const VectorField<2>& u0
    ) const;

    /**
     * Propagate the ice thickness forward in time using the current velocity
     * and accumulation rate.
     */
    Field<2> prognostic_solve(
      const double dt,
      const Field<2>& thickness,
      const Field<2>& accumulation,
      const VectorField<2>& u
    ) const;

    /**
     * Solve the linearization of the diagnostic equation around a velocity u0
     * with a given VectorField as right-hand side
     */
    VectorField<2> adjoint_solve(
      const Field<2>& h,
      const Field<2>& beta,
      const VectorField<2>& u0,
      const VectorField<2>& f
    ) const;
  };

}


#endif
