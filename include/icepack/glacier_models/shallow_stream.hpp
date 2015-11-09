
#ifndef ICEPACK_SHALLOW_STREAM_HPP
#define ICEPACK_SHALLOW_STREAM_HPP

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <icepack/physical_constants.hpp>
#include <icepack/field.hpp>


namespace icepack
{
  using dealii::FE_Q;
  using dealii::FESystem;

  /**
   * This class solves the shallow stream model of glacier flow, appropriate
   * for ice streams and ice shelves which flow with little to no vertical
   * shear.
   */
  class ShallowStream
  {
  public:
    using Velocity = VectorField<2>;
    using Thickness = Field<2>;

    /**
     * Construct a model object, which consists of the geometry and the order
     * of the finite element expansion.
     */
    ShallowStream(
      const Triangulation<2>& triangulation,
      const unsigned int polynomial_order
    );

    ~ShallowStream();

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
     * Compute the ice velocity from the thickness and friction coefficient.
     */
    Velocity diagnostic_solve(
      const Field<2>& surface,
      const Thickness& thickness,
      const Field<2>& beta,
      const Velocity& u0
    );

    /**
     * Propagate the ice thickness forward in time using the current velocity
     * and accumulation rate.
     */
    Thickness prognostic_solve(
      const double dt,
      const Thickness& thickness,
      const Field<2>& accumulation,
      const Velocity& u
    );

    /**
     * Solve the linearization of the diagnostic equation around a velocity u0
     * with a given VectorField as right-hand side
     */
    VectorField<2> adjoint_solve(
      const Field<2>& h,
      const Field<2>& beta,
      const Field<2>& u0,
      const VectorField<2>& f
    );


    /**
     * Accessors
     */
    const Triangulation<2>& get_triangulation() const;

    const FE_Q<2>& get_scalar_fe() const;
    const FESystem<2>& get_vector_fe() const;

    const DoFHandler<2>& get_scalar_dof_handler() const;
    const DoFHandler<2>& get_vector_dof_handler() const;


  protected:
    const Triangulation<2>& triangulation;

    const FE_Q<2> scalar_finite_element;
    const FESystem<2> vector_finite_element;

    DoFHandler<2> scalar_dof_handler;
    DoFHandler<2> vector_dof_handler;
  };

}


#endif
