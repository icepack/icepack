
#ifndef ICEPACK_FIELD_INTERPOLATE_HPP
#define ICEPACK_FIELD_INTERPOLATE_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <icepack/field/field_type.hpp>

namespace icepack {

  using dealii::Function;
  using dealii::TensorFunction;
  using dealii::VectorFunctionFromTensorFunction;

  /**
   * Given a dealii::Function or TensorFunction object, return the function's
   * finite element interpolation as a Field or VectorField respectively.
   */
  template <int dim>
  Field<dim> interpolate(
    const Discretization<dim>& discretization,
    const Function<dim>& phi
  )
  {
    Field<dim> psi(discretization);
    dealii::VectorTools::interpolate(
      psi.get_dof_handler(), phi, psi.get_coefficients()
    );
    return psi;
  }


  /**
   * Overload of `interpolate` for VectorField
   */
  template <int dim>
  VectorField<dim> interpolate(
    const Discretization<dim>& discretization,
    const TensorFunction<1, dim>& phi
  )
  {
    VectorField<dim> psi(discretization);
    const VectorFunctionFromTensorFunction<dim> vphi(phi);
    dealii::VectorTools::interpolate(
      psi.get_dof_handler(), vphi, psi.get_coefficients()
    );
    return psi;
  }


  /**
   * Create a map of degree-of-freedom indices to numbers describing how to
   * create a field with the same boundary values as `phi` on boundary vertices
   * with the id `boundary_id`
   */
  template <int rank, int dim>
  std::map<dealii::types::global_dof_index, double>
  interpolate_boundary_values(
    const FieldType<rank, dim>& phi,
    const unsigned int boundary_id = 0
  )
  {
    auto boundary_values =
      phi.get_field_discretization().zero_boundary_values(boundary_id);

    const Vector<double>& Phi = phi.get_coefficients();
    for (auto& p: boundary_values) p.second = Phi(p.first);

    return boundary_values;
  }


}

#endif
