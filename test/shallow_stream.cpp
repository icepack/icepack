
#include <deal.II/grid/grid_generator.h>

#include <icepack/glacier_models/shallow_stream.hpp>

using namespace dealii;
using namespace icepack;

const double length = 2000.0;
const double width = 500.0;

int main()
{
  Triangulation<2> triangulation;
  const Point<2> p1(0.0, 0.0), p2(length, width);
  GridGenerator::hyper_rectangle(triangulation, p1, p2);

  ShallowStream ssa(triangulation, 1);

  return 0;
}
