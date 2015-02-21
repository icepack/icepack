
#include "read_gridded_data.hpp"
#include <fstream>
#include <vector>

using dealii::Table;

GridData readQgis(const std::string& filename)
{
  unsigned int nx, ny;
  double x0, y0, dx, dy, missing;
  std::string dummy;

  std::ifstream fid(filename);
  fid >> dummy >> nx >> dummy >> ny;
  fid >> dummy >> x0 >> dummy >> y0;
  fid >> dummy >> dx >> dummy >> missing;
  dy = dx;

  std::vector<double> x(nx);
  std::vector<double> y(ny);
  Table<2, double> table(nx, ny);

  for (unsigned int i = 0; i < ny; ++i) y[i] = y0 + i * dy;
  for (unsigned int j = 0; j < nx; ++j) x[j] = x0 + j * dx;

  std::array<std::vector<double>, 2> coordinate_values = {x, y};

  for (unsigned int i = 0; i < ny; ++i) {
    for (unsigned int j = 0; j < nx; ++j) {
      fid >> table[j][ny - i - 1];
    }
  }

  fid.close();

  return GridData (coordinate_values, table);
}
