
// deal.II includes
#include <deal.II/base/function.h>

// our includes
#include "read_gridded_data.hpp"

// C++ includes
#include <fstream>
#include <iostream>


size_t nx = 11, ny = 6;
double dx = 0.2, dy = dx;


using dealii::Point;

void generateExampleArcAsciiGrid(const std::string& filename)
{
  double x0 = 0.0, y0 = 0.0;
  double missing = -9999.0;

  std::ofstream fid(filename);

  fid << "ncols          " << nx << std::endl;
  fid << "nrows          " << ny << std::endl;
  fid << "xllcorner      " << x0 << std::endl;
  fid << "yllcorner      " << y0 << std::endl;
  fid << "cellsize       " << dx << std::endl;
  fid << "NODATA_value   " << missing << std::endl;

  double x, y, z;
  for (size_t i = ny; i > 0; --i) {
    y = y0 + (i - 1) * dy;

    for (size_t j = 0; j < nx; ++j) {
      x = x0 + j * dx;
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

  double x0 = 0.0, y0 = 0.0;

  double x, y;
  double w, z;
  Point<2> p;
  for (size_t i = ny; i > 0; --i) {
    y = y0 + (i - 1) * dy;
    for (size_t j = 0; j < nx; ++j) {
      x = x0 + j * dx;

      p = {x, y};

      z = 1 + x * y;
      w = example_data.value(p, 0);

      if (fabs(w - z) > 1.0e-12) {
        std::cout << "Reading Arc data failed." << std::endl;
        std::cout << "Correct value: " << z << std::endl;
        std::cout << "Data read:     " << w << std::endl;
        return 1;
      }
    }
  }

  double xmin = example_data.xmin(),
         xmax = example_data.xmax(),
         ymin = example_data.ymin(),
         ymax = example_data.ymax();

  if (xmin != x0 or xmax != x0 + nx * dx or
      ymin != y0 or ymax != y0 + ny * dy) {
    std::cout << "Did not correctly record spatial extent of data." << std::endl;
    return 1;
  }

  if (verbose) std::cout << "Reading Arc grid data worked!" << std::endl;
  return 0;
}
