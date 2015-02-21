
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
    }

    fid << std::endl;
  }

  fid.close();
}



int main () {

  bool verbose = true;

  std::string filename = "example_qgis_file.txt";
  generateExampleQgisFile(filename);
  GridData example_data = readQgis(filename);

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
        std::cout << "Reading QGIS data failed." << std::endl;
        std::cout << "Correct value: " << z << std::endl;
        std::cout << "Data read:     " << w << std::endl;
        return 1;
      }
    }
  }

  if (verbose) std::cout << "Reading QGIS data worked!" << std::endl;

  return 0;
}
