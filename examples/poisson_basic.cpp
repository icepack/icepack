
#include "poisson.hpp"
#include "read_grid.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using dealii::ConstantFunction;
using dealii::Function;
using dealii::Triangulation;


template <int dim>
class Coefficient : public Function<dim>
{
public:
  Coefficient () : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const {
    return 1.0 + exp(-p[0]*p[0] - p[1]*p[1]);
  }
};


int main(int argc, char **argv)
{
  ScalarFunctionFromFunctionObject<2> coeff (
    [](const Point<2>& p)
    {
      return 1.0 + exp(-p[0]*p[0]-p[1]*p[1]);
    }
  );

  ConstantFunction<2, double> rhs(1.0);
  Triangulation<2> tri = read_gmsh_grid<2>(argv[1]);

  PoissonProblem<2> pp(tri, coeff, rhs);
  pp.run();
  pp.output("solution");

  return 1;
}
