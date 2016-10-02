
#ifndef ICEPACK_DEPTH_AVERAGED_MODEL
#define ICEPACK_DEPTH_AVERAGED_MODEL

#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>

#include <icepack/field.hpp>
#include <icepack/physics/viscosity.hpp>

namespace icepack {

  using dealii::Triangulation;
  using dealii::Function;

  /**
   * This class contains methods and data common to all depth-averaged models
   * of glacier flow. For example, every depth-averaged model consists of a
   * diagnostic equation, which dictates the depth-averaged velocity, and a
   * prognostic equation, which dictates the evolution of the ice thickness
   * field given a current velocity.
   * DepthAveragedModel consists only of common components that other glacier
   * models use; it is not useful on its own.
   */
  class DepthAveragedModel
  {
  public:
    /**
     * Construct a DepthAveragedModel for a given geometry and degree of the
     * finite element basis functions.
     */
    DepthAveragedModel(
      const Triangulation<2>& triangulation,
      unsigned int polynomial_order,
      double newton_tolerance = 1.0e-6,
      double picard_tolerance = 1.0e-2,
      unsigned int max_iterations = 20
    );

    /**
     * Given some observed data, represented by a `dealii::Function object``,
     * compute the finite element interpolation of the data using the basis
     * for this model.
     */
    Field<2> interpolate(const Function<2>& phi) const;

    /**
     * Given observed vector data, represented by a `dealii::TensorFunction`
     * object, compute the finite element interpolation of the data using
     * the basis for this model.
     */
    VectorField<2> interpolate(const TensorFunction<1, 2>& phi) const;


    /**
     * Given two observed scalar fields, represented by `dealii::Function`
     * objects, compute a finite element-discretized vector field with these
     * functions as coordinates
     */
    VectorField<2>
    interpolate(const Function<2>& phi0, const Function<2>& phi1) const;


    // TODO add warning about how we use an overly diffusive approximation
    /**
     * Compute the rate of change of the ice thickness using the current
     * accumulation rate and depth-averaged velocity.
     */
    Field<2> dh_dt(
      const Field<2>& thickness,
      const Field<2>& accumulation,
      const VectorField<2>& u
    ) const;


    /**
     * Propagate the ice thickness forward in time using the current
     * accumulation rate and depth-averaged velocity.
     */
    Field<2> prognostic_solve(
      const double dt,
      const Field<2>& thickness,
      const Field<2>& accumulation,
      const VectorField<2>& u
    ) const;


    /*
     * Accessors
     */

    const Discretization<2>& get_discretization() const;
    const Triangulation<2>& get_triangulation() const;

    /**
     * Function object for computing the constitutive tensor
     */
    const ConstitutiveTensor constitutive_tensor;

  protected:
    const Discretization<2> discretization;
    const double newton_tolerance;
    const double picard_tolerance;
    const unsigned int max_iterations;
  };
}

#endif
