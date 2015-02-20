
// deal.II includes
#include <deal.II/base/function.h>

// our includes
#include "read_gridded_data.hpp"

// C++ includes
#include <fstream>
#include <iostream>

using dealii::Point;

void generateExampleQgisFile(const std::string& filename) {

  size_t nx = 51, ny = 21;
  double dx = 0.25, x0 = 1.0, y0 = 2.0, dy = dx;
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
      z = x * y;
      fid << z << " ";
    }

    fid << std::endl;
  }

  fid.close();
}



int main () {

  std::string filename = "test_read_gridded_data_example.txt";
  generateExampleQgisFile(filename);
  GridData example_data = readQgis(filename);

  size_t nx = 51, ny = 21;
  double dx = 0.25, x0 = 1.0, y0 = 2.0, dy = dx;

  double x, y, z;
  Point<2> p;
  for (size_t i = 0; i < ny; ++i) {
    y = y0 + i * dy;
    for (size_t j = 0; j < nx; ++j) {
      x = x0 + j * dx;
      z = x * y;
      p = {x, y};
      if ( fabs(example_data.value(p) - z) > 1.0e-12 ) {
        std::cout << "Uh-oh, wrong function value!" << std::endl;
        std::cout << i << ", " << j << std::endl;
        std::cout << example_data.value(p) << ", " << z  << std::endl;
        return 1;
      }
    }
  }

  return 0;
}
