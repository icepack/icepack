
#ifndef ICEPACK_DEPTH_AVERAGED_MODEL
#define ICEPACK_DEPTH_AVERAGED_MODEL

#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>

#include <icepack/glacier_models/pde_skeleton.hpp>

namespace icepack {

  using dealii::Triangulation;
  using dealii::Function;

  class DepthAveragedModel
  {
  public:
    /**
     * Given some observed data, represented by a dealii::Function object,
     * compute the finite element interpolation of the data using the basis
     * for this model.
     */
    Field<2> interpolate(const Function<2>& phi) const;

    /**
     * Given observed vector data, represented by a dealii::TensorFunction
     * object, compute the finite element interpolation of the data using
     * the basis for this model.
     */
    VectorField<2> interpolate(const TensorFunction<1, 2>& phi) const;


    /*
     * Accessors
     */
    const Triangulation<2>& get_triangulation() const;
    const ScalarPDESkeleton<2>& get_scalar_pde_skeleton() const;
    const VectorPDESkeleton<2>& get_vector_pde_skeleton() const;

  protected:
    DepthAveragedModel(
      const Triangulation<2>& triangulation,
      const unsigned int polynomial_order
    );

    const Triangulation<2>& triangulation;

    const ScalarPDESkeleton<2> scalar_pde;
    const VectorPDESkeleton<2> vector_pde;

  private:

  };
}

#endif
