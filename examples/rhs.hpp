
#ifndef RHS_HPP
#define RHS_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

using dealii::Function;
using dealii::Point;
using dealii::Vector;
using dealii::Tensor;

class SurfaceElevation : public Function<2>
{
public:
  SurfaceElevation ();

  virtual double value(const Point<2>& x,
                       const unsigned int component = 0) const;
  virtual Tensor<1, 2> gradient(const Point<2>& x,
                                const unsigned int component = 0) const;
  virtual void gradient_list(const std::vector< Point<2> >& points,
                             std::vector< Tensor<1, 2> >& gradients,
                             const unsigned int component = 0) const;
};


SurfaceElevation::SurfaceElevation ()
  :
  Function<2> (2)
{}


inline
double SurfaceElevation::value(const Point<2>& x,
                               const unsigned int component) const
{
  Assert(component = 0, ExcNotImplemented());

  return exp(-x.square());
}


inline
Tensor<1, 2> SurfaceElevation::gradient(const Point<2>& x,
                                        const unsigned int component) const
{
  Tensor<1, 2> v;
  v[0] = -2 * x[0] * exp(-x.square());
  v[1] = -2 * x[1] * exp(-x.square());
  return v;
}


void SurfaceElevation::gradient_list(const std::vector< Point<2> >& points,
                                     std::vector< Tensor<1, 2> >& gradients,
                                     const unsigned int component) const
{
  Assert(value_list.size() == points.size(),
         ExcDimensionMismatch(value_list.size(), points.size()));

  const unsigned int n_points = points.size();
  for (unsigned int p = 0; p < n_points; ++p)
    gradients[p] = SurfaceElevation::gradient(points[p]);
}


#endif
