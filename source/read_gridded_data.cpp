
#include "read_gridded_data.hpp"
#include <fstream>

using dealii::Table;

GridData readQgis(const std::string& filename)
{
  unsigned int nx, ny;
  double x0, y0, dx, dy, missing, q;
  std::string dummy;

  std::ifstream fid(filename);
  fid >> dummy >> nx >> dummy >> ny;
  fid >> dummy >> x0 >> dummy >> y0;
  fid >> dummy >> dx >> dummy >> missing;
  dy = dx;

  std::array<std::pair<double, double>, 2> endpoints;
  endpoints[0] = std::make_pair (x0, x0 + nx * dx);
  endpoints[1] = std::make_pair (y0, y0 + ny * dy);

  std::array<unsigned int, 2> n_intervals = {nx - 1, ny - 1};

  std::vector<double> data;

  for (size_t i = 0; i < ny; ++i)
    for (size_t j = 0; j < nx; ++j)
      {
        fid >> q;
        data.push_back (q);
      }

  Table<2, double> table (nx, ny, data.begin());

  return GridData (endpoints, n_intervals, table);
}
