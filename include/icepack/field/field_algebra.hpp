
#ifndef ICEPACK_FIELD_ALGEBRA_HPP
#define ICEPACK_FIELD_ALGEBRA_HPP

#include <icepack/field/field.hpp>

namespace icepack {

  using dealii::ExcInternalError;

  template <int rank, int dim>
  FieldType<rank, dim>&
  operator *=(FieldType<rank, dim>& phi, const double alpha)
  {
    phi.get_coefficients() *= alpha;
    return phi;
  }


  template <int rank, int dim>
  FieldType<rank, dim>&
  operator /=(FieldType<rank, dim>& phi, const double alpha)
  {
    phi.get_coefficients() *= 1.0/alpha;
    return phi;
  }


  template <int rank, int dim>
  FieldType<rank, dim>&
  operator +=(FieldType<rank, dim>& phi, const FieldType<rank, dim>& psi)
  {
    // TODO: handle the case where the discretizations aren't identical
    Assert(phi.has_same_discretization(psi), ExcInternalError());

    phi.get_coefficients().add(1.0, psi.get_coefficients());
    return phi;
  }


  template <int rank, int dim>
  FieldType<rank, dim>&
  operator -=(FieldType<rank, dim>& phi, const FieldType<rank, dim>& psi)
  {
    // TODO: handle the case where the discretizations aren't identical
    Assert(phi.has_same_discretization(psi), ExcInternalError());

    phi.get_coefficients().add(-1.0, psi.get_coefficients());
    return phi;
  }

} // End of namespace icepack

#endif
