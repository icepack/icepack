
#ifndef ICEPACK_FIELD_ALGEBRA_HPP
#define ICEPACK_FIELD_ALGEBRA_HPP

#include <icepack/field/field_type.hpp>

namespace icepack {

  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  operator *=(FieldType<rank, dim, duality>& phi, const double alpha)
  {
    phi.get_coefficients() *= alpha;
    return phi;
  }


  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  operator /=(FieldType<rank, dim, duality>& phi, const double alpha)
  {
    phi.get_coefficients() *= 1.0/alpha;
    return phi;
  }


  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  operator +=(FieldType<rank, dim, duality>& phi,
              const FieldType<rank, dim, duality>& psi)
  {
    Assert(have_same_discretization(phi, psi), ExcInternalError());

    phi.get_coefficients().add(1.0, psi.get_coefficients());
    return phi;
  }


  template <int rank, int dim, Duality duality, class Expr>
  FieldType<rank, dim, duality>&
  operator +=(FieldType<rank, dim, duality>& phi,
              const FieldExpr<rank, dim, duality, Expr>& expr)
  {
    Assert(have_same_discretization(phi, expr), ExcInternalError());

    Vector<double>& Phi = phi.get_coefficients();
    for (unsigned int i = 0; i < Phi.size(); ++i)
      Phi(i) += expr.coefficient(i);

    return phi;
  }


  template <int rank, int dim, Duality duality>
  FieldType<rank, dim, duality>&
  operator -=(FieldType<rank, dim, duality>& phi,
              const FieldType<rank, dim, duality>& psi)
  {
    Assert(have_same_discretization(phi, psi), ExcInternalError());

    phi.get_coefficients().add(-1.0, psi.get_coefficients());
    return phi;
  }


  template <int rank, int dim, Duality duality, class Expr>
  FieldType<rank, dim, duality>&
  operator -=(FieldType<rank, dim, duality>& phi,
              const FieldExpr<rank, dim, duality, Expr>& expr)
  {
    Assert(have_same_discretization(phi, expr), ExcInternalError());

    Vector<double>& Phi = phi.get_coefficients();
    for (unsigned int i = 0; i < Phi.size(); ++i)
      Phi(i) -= expr.coefficient(i);

    return phi;
  }



  /* ------------------------------
   * Algebraic expression templates
   * ------------------------------ */

  template <int rank, int dim, Duality duality, class Expr>
  class ScalarMultiplyExpr :
    public FieldExpr<rank, dim, duality,
                     ScalarMultiplyExpr<rank, dim, duality, Expr> >
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

    const Discretization<dim>& get_discretization() const
    {
      return expr.get_discretization();
    }

  protected:
    const double alpha;
    const Expr& expr;
  };


  template <int rank, int dim, Duality duality, class Expr>
  ScalarMultiplyExpr<rank, dim, duality, Expr>
  operator*(const double alpha, const FieldExpr<rank, dim, duality, Expr>& expr)
  {
    return ScalarMultiplyExpr<rank, dim, duality, Expr>(alpha, expr);
  }


  template <int rank, int dim, Duality duality, class Expr>
  ScalarMultiplyExpr<rank, dim, duality, Expr>
  operator/(const FieldExpr<rank, dim, duality, Expr>& expr, const double alpha)
  {
    return ScalarMultiplyExpr<rank, dim, duality, Expr>(1.0/alpha, expr);
  }


  template <int rank, int dim, Duality duality, class Expr>
  ScalarMultiplyExpr<rank, dim, duality, Expr>
  operator-(const FieldExpr<rank, dim, duality, Expr>& expr)
  {
    return ScalarMultiplyExpr<rank, dim, duality, Expr>(-1.0, expr);
  }


  template <int rank, int dim, Duality duality, class Expr1, class Expr2>
  class AddExpr :
    public FieldExpr<rank, dim, duality,
                     AddExpr<rank, dim, duality, Expr1, Expr2> >
  {
  public:
    AddExpr(const Expr1& expr1, const Expr2& expr2)
      :
      expr1(expr1),
      expr2(expr2)
    {
      Assert(have_same_discretization(expr1, expr2), ExcInternalError());
    }

    double coefficient(const size_t i) const
    {
      return expr1.coefficient(i) + expr2.coefficient(i);
    }

    const Discretization<dim>& get_discretization() const
    {
      // Since we've asserted that both `expr1` and `expr2` have identical (in
      // the sense of pointer equality) discretizations, we can just return the
      // first expression's discretization.
      return expr1.get_discretization();
    }

  protected:
    const Expr1& expr1;
    const Expr2& expr2;
  };


  template <int rank, int dim, Duality duality, class Expr1, class Expr2>
  AddExpr<rank, dim, duality, Expr1, Expr2>
  operator+(const FieldExpr<rank, dim, duality, Expr1>& expr1,
            const FieldExpr<rank, dim, duality, Expr2>& expr2)
  {
    return AddExpr<rank, dim, duality, Expr1, Expr2>(expr1, expr2);
  }


  template <int rank, int dim, Duality duality, class Expr1, class Expr2>
  class SubtractExpr :
    public FieldExpr<rank, dim, duality,
                     SubtractExpr<rank, dim, duality, Expr1, Expr2> >
  {
  public:
    SubtractExpr(const Expr1& expr1, const Expr2& expr2)
      :
      expr1(expr1),
      expr2(expr2)
    {
      Assert(have_same_discretization(expr1, expr2), ExcInternalError());
    }

    double coefficient(const size_t i) const
    {
      return expr1.coefficient(i) - expr2.coefficient(i);
    }

    const Discretization<dim>& get_discretization() const
    {
      return expr1.get_discretization();
    }

  protected:
    const Expr1& expr1;
    const Expr2& expr2;
  };


  template <int rank, int dim, Duality duality, class Expr1, class Expr2>
  SubtractExpr<rank, dim, duality, Expr1, Expr2>
  operator-(const FieldExpr<rank, dim, duality, Expr1>& expr1,
            const FieldExpr<rank, dim, duality, Expr2>& expr2)
  {
    return SubtractExpr<rank, dim, duality, Expr1, Expr2>(expr1, expr2);
  }

} // End of namespace icepack

#endif
