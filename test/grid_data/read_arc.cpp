
#include <fstream>
#include <iostream>
#include <deal.II/base/function.h>
#include <icepack/grid_data.hpp>
#include "../testing.hpp"

using namespace icepack;
using dealii::Point;


constexpr size_t nx = 11, ny = 6;
constexpr double xo = 0.0, yo = 0.0;
constexpr double dx = 0.2, dy = dx;


void generateExampleArcAsciiGrid(const std::string& filename)
{
  double missing = -9999.0;

  std::ofstream fid(filename);

  fid << "ncols          " << nx << std::endl;
  fid << "nrows          " << ny << std::endl;
  fid << "xllcorner      " << xo << std::endl;
  fid << "yllcorner      " << yo << std::endl;
  fid << "cellsize       " << dx << std::endl;
  fid << "NODATA_value   " << missing << std::endl;

  double x, y, z;
  for (size_t i = ny; i > 0; --i) {
    y = yo + (i - 1) * dy;

    for (size_t j = 0; j < nx; ++j) {
      x = xo + j * dx;
      z = 1 + x * y;
      fid << z << " ";
    }

    fid << std::endl;
  }

  fid.close();
}



int main (int argc, char **argv)
{
  bool verbose = false;
  if (strcmp(argv[argc-1], "-v") == 0) verbose = true;

  std::string filename = "example_arc_file.txt";
  generateExampleArcAsciiGrid(filename);
  GridData example_data = readArcAsciiGrid(filename);

  double xmin = example_data.xrange[0],
         xmax = example_data.xrange[1],
         ymin = example_data.yrange[0],
         ymax = example_data.yrange[1];

  check (xmin == xo & xmax == xo + (nx - 1) * dx &
         ymin == yo & ymax == yo + (ny - 1) * dy);

  double x, y;
  double w, z;
  Point<2> p;
  for (size_t i = ny; i > 0; --i) {
    y = yo + (i - 1) * dy;
    for (size_t j = 0; j < nx; ++j) {
      x = xo + j * dx;

      p = {x, y};

      z = 1 + x * y;
      w = example_data.value(p, 0);

      check(fabs(w - z) < 1.0e-12);
    }
  }

  if (verbose) std::cout << "Reading Arc grid data worked!" << std::endl;
  return 0;
}
