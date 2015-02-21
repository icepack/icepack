
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

void generateExampleQgisFile(const std::string& filename) {

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
      std::cout << z << " ";
    }

    std::cout << std::endl;
    fid << std::endl;
  }

  std::cout << std::endl;
  fid.close();
}



int main () {

  std::string filename = "test_read_gridded_data_example.txt";
  generateExampleQgisFile(filename);
  GridData example_data = readQgis(filename);

  double x0 = 0.0, y0 = 0.0;

  double x, y, z;
  Point<2> p;
  for (size_t i = 0; i < ny; ++i) {
    y = y0 + i * dy;
    for (size_t j = 0; j < nx; ++j) {
      x = x0 + j * dx;
      z = 1 + x * y;
      p = {x, y};
      std::cout << example_data.value(p, 0) << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
