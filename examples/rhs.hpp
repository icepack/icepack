
#ifndef RHS_HPP
#define RHS_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

using dealii::Function;
using dealii::Point;
using dealii::Vector;
using dealii::Tensor;

template <int dim>
class RightHandSide :  public Function<dim>
{
public:
  RightHandSide ();

  virtual double value(const Point<dim>& x,
                       const unsigned int component = 0) const;
  virtual Tensor<1, dim> gradient(const Point<dim>& x,
                                  const unsigned int component = 0) const;
  virtual void gradient_list(const std::vector< Point<dim> >& points,
                             std::vector< Tensor<1, dim> >& gradients,
                             const unsigned int component = 0) const;
};


template <int dim>
RightHandSide<dim>::RightHandSide ()
  :
  Function<dim> (dim)
{}


template <int dim>
inline
double RightHandSide<dim>::value(const Point<dim>& x,
                                 const unsigned int component) const
{
  Assert(dim >= 2, ExcNotImplemented());
  Assert(component = 0, ExcNotImplemented());

  return exp(-x.square());
}


template <int dim>
inline
Tensor<1, dim> RightHandSide<dim>::gradient(const Point<dim>& x,
                                            const unsigned int component) const
{
  Tensor<1, dim> v;
  for (unsigned int i = 0; i < dim; ++i) {
    v[i] = -2 * x[i] * exp(-x.square());
  }
  return v;
}


template <int dim>
void RightHandSide<dim>::gradient_list(const std::vector< Point<dim> >& points,
                                       std::vector< Tensor<1, dim> >& gradients,
                                       const unsigned int component) const
{
  Assert(value_list.size() == points.size(),
         ExcDimensionMismatch(value_list.size(), points.size()));

  const unsigned int n_points = points.size();
  for (unsigned int p = 0; p < n_points; ++p)
    gradients[p] = RightHandSide<dim>::gradient(points[p]);
}


#endif
