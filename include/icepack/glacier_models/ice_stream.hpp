
#ifndef ICEPACK_ICE_STREAM_HPP
#define ICEPACK_ICE_STREAM_HPP

#include <icepack/physical_constants.hpp>
#include <icepack/field.hpp>
#include <icepack/glacier_models/pde_skeleton.hpp>


namespace icepack
{

  /**
   * This class solves the shallow stream model of glacier flow, appropriate
   * for ice streams and ice shelves which flow with little to no vertical
   * shear.
   */
  class IceStream
  {
  public:

    /**
     * Construct a model object, which consists of the geometry and the order
     * of the finite element expansion.
     */
    IceStream(
      const Triangulation<2>& triangulation,
      const unsigned int polynomial_order
    );


    /**
     * Given some observed data, represented by a dealii::Function object,
     * compute the finite element interpolation of the data using the finite
     * element basis for this model.
     */
    Field<2> interpolate(const Function<2>& phi) const;

    /**
     * Given observed data for a vector field, represented by a dealii::
     * TensorFunction object, compute the finite element interpolation of the
     * data using the finite element basis for this model.
     */
    VectorField<2> interpolate(const TensorFunction<1, 2>& f) const;


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


    /*
     * Accessors
     */
    const Triangulation<2>& get_triangulation() const;
    const ScalarPDESkeleton<2>& get_scalar_pde_skeleton() const;
    const VectorPDESkeleton<2>& get_vector_pde_skeleton() const;

  protected:
    const Triangulation<2>& triangulation;

    const ScalarPDESkeleton<2> scalar_pde;
    const VectorPDESkeleton<2> vector_pde;
  };

}


#endif
