
#ifndef RHS_HPP
#define RHS_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>

using dealii::Function;
using dealii::TensorFunction;
using dealii::Point;
using dealii::Vector;
using dealii::Tensor;
using dealii::StandardExceptions::ExcNotImplemented;
using dealii::StandardExceptions::ExcDimensionMismatch;


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
  Assert(component == 0, ExcNotImplemented());

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
  Assert(gradients.size() == points.size(),
         ExcDimensionMismatch(gradients.size(), points.size()));

  const unsigned int n_points = points.size();
  for (unsigned int p = 0; p < n_points; ++p)
    gradients[p] = SurfaceElevation::gradient(points[p]);
}


class BedElevation : public Function<2>
{
public:
  BedElevation () : Function<2>() {}
  virtual double value (const Point<2>& x,
                        const unsigned int component = 0) const;
  virtual void value_list (const std::vector<Point<2> >& points,
                           std::vector<double>&          values,
                           const unsigned int            component = 0) const;
};


double BedElevation::value (const Point<2>& x,
                            const unsigned int) const
{
  return -1.0 - exp(-x.square());
}


void BedElevation::value_list (const std::vector<Point<2> >& points,
                               std::vector<double>&          values,
                               const unsigned int            component) const
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));
  const unsigned int n_points = points.size();

  for (unsigned int i = 0; i < n_points; ++i)
    values[i] = -1.0 - exp(-points[i].square());
}


class BoundaryVelocity : public TensorFunction<1, 2>
{
public:
  BoundaryVelocity () : TensorFunction<1, 2>() {}
  virtual Tensor<1, 2> value (const Point<2>& x) const;
};


Tensor<1, 2> BoundaryVelocity::value(const Point<2>& x) const
{
  Tensor<1, 2> v;
  v[0] = 0.25 * (1.0 - x[1]) * (1.0 + x[1]);
  v[1] = 0.0;
  return v;
}



#endif
