
#include "poisson.hpp"
#include "read_grid.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using dealii::ConstantFunction;
using dealii::Triangulation;

int main(int argc, char **argv)
{
  ConstantFunction<2, double> coeff(1.0);
  ConstantFunction<2, double> rhs(1.0);
  Triangulation<2> tri = read_gmsh_grid<2>(argv[1]);

  PoissonProblem<2> pp(tri, coeff, rhs);
  pp.run();

  return 1;
}
