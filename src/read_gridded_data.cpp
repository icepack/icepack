
#include "endianness.hpp"
#include "read_gridded_data.hpp"

#include <fstream>
#include <sstream>
#include <vector>

using dealii::Table;

GridData readArcAsciiGrid(const std::string& filename)
/**
 * Read a gridded data set stored in the ESRI Arc/Info ASCII grid format. See
 *   http://en.wikipedia.org/wiki/Esri_grid
 * for format specification and more info.
 */
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

  std::array<std::vector<double>, 2> coordinate_values = {{x, y}};

  for (unsigned int i = 0; i < ny; ++i)
    for (unsigned int j = 0; j < nx; ++j)
      fid >> table[j][ny - i - 1];

  fid.close();

  return GridData (coordinate_values, table);
}


static void readGeoDatInfo(const std::string& filename,
                 unsigned int& nx,
                 unsigned int& ny,
                 double& dx,
                 double& dy,
                 double& xo,
                 double& yo)
/**
 * Parse the grid size and resolution information contained in the file
 *     <filename>.geodat
 * This is a helper function for readGeoDat (below).
 */
{
  std::ifstream geoDatInfoFile(filename + ".geodat");
  std::string line;

  while (not (std::istringstream(line) >> nx >> ny))
    std::getline(geoDatInfoFile, line);

  while (not (std::istringstream(line) >> dx >> dy))
    std::getline(geoDatInfoFile, line);

  while (not (std::istringstream(line) >> xo >> yo))
    std::getline(geoDatInfoFile, line);

  xo = xo * 1000.0;
  yo = yo * 1000.0;

  geoDatInfoFile.close();
}


GridData readGeoDat(const std::string& filename)
/**
 * Read gridded data from the format used in IR Joughin's ice velocity
 * and elevation data sets from the National Snow and Ice Data Center
 */
{
  unsigned int nx, ny;
  double dx, dy, xo, yo;
  readGeoDatInfo(filename, nx, ny, dx, dy, xo, yo);

  std::ifstream geoDatFile(filename, std::ios::in | std::ios::binary);

  float q;
  unsigned char temp[sizeof(float)];
  std::vector<float> vals;

  while (not geoDatFile.eof()) {
    // If a float isn't 4 bytes on your system, then may God have
    // mercy on your soul.
    geoDatFile.read(reinterpret_cast<char*>(temp), sizeof(float));
    q = ntohx(reinterpret_cast<float&>(temp));
    vals.push_back(q);
  }

  geoDatFile.close();

  std::vector<double> x(nx);
  std::vector<double> y(ny);
  Table<2, double> table(nx, ny);

  for (unsigned int i = 0; i < ny; ++i) y[i] = yo + i * dy;
  for (unsigned int j = 0; j < nx; ++j) x[j] = xo + j * dx;

  std::array<std::vector<double>, 2> coordinate_values = {{x, y}};

  for (unsigned int i = 0; i < ny; ++i)
    for (unsigned int j = 0; j < nx; ++j)
      table[j][i] = vals[ny * j + i];

  return GridData(coordinate_values, table);
}
