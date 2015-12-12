
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
    /**
     * Construct a model object for a given geometry and finite element basis.
     */
    IceShelf(
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
    VectorField<2> driving_stress(const Field<2>& thickness) const;

    /**
     * Compute the residual of a candidate solution to the diagnostic equation.
     * This vector is used to solve the system by Newton's method.
     */
    VectorField<2> residual(
      const Field<2>& thickness,
      const VectorField<2>& u,
      const VectorField<2>& tau_d
    ) const;

    /**
     * Compute the ice velocity from the thickness.
     */
    VectorField<2> diagnostic_solve(
      const Field<2>& thickness,
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
      const Field<2>& h,
      const VectorField<2>& u0,
      const VectorField<2>& f
    ) const;
  };

}

#endif
