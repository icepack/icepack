
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


constexpr double radius = 5e3;
constexpr double slope = 0.01;
constexpr double s_max = 100;
constexpr double s_min = 10;


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

  double yp = x[1] - radius, xp = x[0];
  return 20.0 + 2.0 * x[1] / radius
    + 5.0 * exp(-(xp*xp/1.0e4 + yp*yp/4.0e4));
}


inline
Tensor<1, 2> SurfaceElevation::gradient(const Point<2>& x,
                                        const unsigned int component) const
{
  double yp = x[1] - radius, xp = x[0], q = exp(-(xp*xp/1.0e4 + yp*yp/4.0e4));

  Tensor<1, 2> v;
  v[0] = -10.0 * xp/1.0e4 * q;
  v[1] = 2.0/radius - 10.0 * yp/4.0e4 * q;
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
  return -2000.0;
}


void BedElevation::value_list (const std::vector<Point<2> >& points,
                               std::vector<double>&          values,
                               const unsigned int            component) const
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));
  const unsigned int n_points = points.size();

  for (unsigned int i = 0; i < n_points; ++i)
    values[i] = -2000.0;
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
  v[0] = 0.0;
  v[1] = -125.0 * exp(-x[0]*x[0] / 1.0e4); //-100.0;
  return v;
}



#endif
