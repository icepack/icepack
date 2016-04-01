
#ifndef ICEPACK_FIELD_ALGEBRA_HPP
#define ICEPACK_FIELD_ALGEBRA_HPP

#include <icepack/field/field_type.hpp>

namespace icepack {

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


  /* ------------------------------
   * Algebraic expression templates
   * ------------------------------ */

  template <int rank, int dim, class Expr>
  class ScalarMultiplyExpr :
    public FieldExpr<rank, dim, ScalarMultiplyExpr<rank, dim, Expr> >
  {
  public:
    ScalarMultiplyExpr(const double alpha, const Expr& expr)
      :
      alpha(alpha),
      expr(expr)
    {}

    double coefficient(const size_t i) const
    {
      return alpha * expr.coefficient(i);
    }

  protected:
    const double alpha;
    const Expr& expr;
  };


  template <int rank, int dim, class Expr>
  ScalarMultiplyExpr<rank, dim, Expr>
  operator*(const double alpha, const FieldExpr<rank, dim, Expr>& expr)
  {
    return ScalarMultiplyExpr<rank, dim, Expr>(alpha, expr);
  }


  template <int rank, int dim, class Expr>
  ScalarMultiplyExpr<rank, dim, Expr>
  operator/(const FieldExpr<rank, dim, Expr>& expr, const double alpha)
  {
    return ScalarMultiplyExpr<rank, dim, Expr>(1.0/alpha, expr);
  }


  template <int rank, int dim, class Expr>
  ScalarMultiplyExpr<rank, dim, Expr>
  operator-(const FieldExpr<rank, dim, Expr>& expr)
  {
    return ScalarMultiplyExpr<rank, dim, Expr>(-1.0, expr);
  }


  template <int rank, int dim, class Expr1, class Expr2>
  class AddExpr :
    public FieldExpr<rank, dim, AddExpr<rank, dim, Expr1, Expr2> >
  {
  public:
    AddExpr(const Expr1& expr1, const Expr2& expr2)
      :
      expr1(expr1),
      expr2(expr2)
    {}

    double coefficient(const size_t i) const
    {
      return expr1.coefficient(i) + expr2.coefficient(i);
    }

  protected:
    const Expr1& expr1;
    const Expr2& expr2;
  };


  template <int rank, int dim, class Expr1, class Expr2>
  AddExpr<rank, dim, Expr1, Expr2>
  operator+(const FieldExpr<rank, dim, Expr1>& expr1,
            const FieldExpr<rank, dim, Expr2>& expr2)
  {
    return AddExpr<rank, dim, Expr1, Expr2>(expr1, expr2);
  }


  template <int rank, int dim, class Expr1, class Expr2>
  class SubtractExpr :
    public FieldExpr<rank, dim, SubtractExpr<rank, dim, Expr1, Expr2> >
  {
  public:
    SubtractExpr(const Expr1& expr1, const Expr2& expr2)
      :
      expr1(expr1),
      expr2(expr2)
    {}

    double coefficient(const size_t i) const
    {
      return expr1.coefficient(i) - expr2.coefficient(i);
    }

  protected:
    const Expr1& expr1;
    const Expr2& expr2;
  };


  template <int rank, int dim, class Expr1, class Expr2>
  SubtractExpr<rank, dim, Expr1, Expr2>
  operator-(const FieldExpr<rank, dim, Expr1>& expr1,
            const FieldExpr<rank, dim, Expr2>& expr2)
  {
    return SubtractExpr<rank, dim, Expr1, Expr2>(expr1, expr2);
  }

} // End of namespace icepack

#endif
