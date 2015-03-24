
#ifndef RHS_HPP
#define RHS_HPP

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

using dealii::Function;
using dealii::Point;
using dealii::Vector;

template <int dim>
class RightHandSide :  public Function<dim>
{
public:
  RightHandSide ();

  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &values) const;

  virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                  std::vector<Vector<double> >   &value_list) const;
};


template <int dim>
RightHandSide<dim>::RightHandSide ()
  :
  Function<dim> (dim)
{}


template <int dim>
inline
void RightHandSide<dim>::vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const
{
  Assert (values.size() == dim,
          ExcDimensionMismatch (values.size(), dim));
  Assert (dim >= 2, ExcNotImplemented());

  Point<dim> point_1, point_2;
  point_1(0) = 0.5;
  point_2(0) = -0.5;

  if (((p-point_1).square() < 0.2*0.2) ||
      ((p-point_2).square() < 0.2*0.2))
    values(0) = 1;
  else
    values(0) = 0;

  if (p.square() < 0.2*0.2)
    values(1) = 1;
  else
    values(1) = 0;
}



template <int dim>
void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                            std::vector<Vector<double> >   &value_list) const
{
  Assert (value_list.size() == points.size(),
          ExcDimensionMismatch (value_list.size(), points.size()));

  const unsigned int n_points = points.size();

  for (unsigned int p=0; p<n_points; ++p)
    RightHandSide<dim>::vector_value (points[p],
                                      value_list[p]);
}

#endif
