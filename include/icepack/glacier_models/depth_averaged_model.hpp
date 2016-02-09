
#ifndef ICEPACK_DEPTH_AVERAGED_MODEL
#define ICEPACK_DEPTH_AVERAGED_MODEL

#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>

#include <icepack/glacier_models/pde_skeleton.hpp>
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

    /**
     * Return a reference to the model geometry.
     */
    const Triangulation<2>& get_triangulation() const;

    /**
     * Return a reference to the data for a scalar PDE with the given model and
     * geometry, such as the prognostic equation.
     */
    const ScalarPDESkeleton<2>& get_scalar_pde_skeleton() const;

    /**
     * Return a reference to the data for a vector PDE with the given model and
     * geometry, such as the diagnostic equation.
     */
    const VectorPDESkeleton<2>& get_vector_pde_skeleton() const;

    /**
     * Function object for computing the constitutive tensor
     */
    const ConstitutiveTensor constitutive_tensor;

  protected:
    /**
     * Construct a DepthAveragedModel for a given geometry and degree of the
     * finite element basis functions. This constructor is not public; it is
     * instead invoked in the constructor of child classes like `IceShelf` or
     * `IceStream`.
     */
    DepthAveragedModel(
      const Triangulation<2>& triangulation,
      const unsigned int polynomial_order
    );

    /**
     * The geometry for the model. This member is a reference to a
     * `dealii::Triangulation` because the model does not own the geometry.
     */
    const Triangulation<2>& triangulation;

    /**
     * The scalar PDE skeleton stores all of the data that will be shared by
     * any scalar PDE over the given geometry regardless of its character or
     * physical meaning, i.e. both the prognostic equation and the heat
     * equation are built on the same scalar PDE skeleton even though they are
     * distinct PDEs of different types.
     */
    const ScalarPDESkeleton<2> scalar_pde;

    /**
     * Store all data shared by systems of PDE over the given geometry.
     */
    const VectorPDESkeleton<2> vector_pde;

  private:

  };
}

#endif
