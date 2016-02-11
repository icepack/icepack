
#include <fstream>
#include <sstream>
#include <vector>

#include <gdal_priv.h>
#include <cpl_conv.h>

#include <icepack/endianness.hpp>
#include <icepack/read_gridded_data.hpp>

using dealii::Table;
using dealii::Functions::InterpolatedTensorProductGridData;

namespace icepack
{

  GridData::GridData(
    const std::array<std::vector<double>, 2>& coordinate_values,
    const Table<2, double>& data_values,
    const double missing
  )
    :
    InterpolatedTensorProductGridData<2>(coordinate_values, data_values),
    xrange{{coordinate_values[0][0], coordinate_values[0].back()}},
    yrange{{coordinate_values[1][0], coordinate_values[1].back()}},
    missing(missing)
  {}


  GridData readArcAsciiGrid(const std::string& filename)
  {
    unsigned int nx, ny;
    double x0, y0, dx, dy, missing;
    std::string dummy;

    std::ifstream file_stream(filename);
    file_stream >> dummy >> nx >> dummy >> ny;
    file_stream >> dummy >> x0 >> dummy >> y0;
    file_stream >> dummy >> dx >> dummy >> missing;
    dy = dx;

    std::vector<double> x(nx);
    std::vector<double> y(ny);
    Table<2, double> table(nx, ny);

    for (unsigned int i = 0; i < ny; ++i) y[i] = y0 + i * dy;
    for (unsigned int j = 0; j < nx; ++j) x[j] = x0 + j * dx;

    std::array<std::vector<double>, 2> coordinate_values = {{x, y}};

    for (unsigned int i = 0; i < ny; ++i)
      for (unsigned int j = 0; j < nx; ++j)
        file_stream >> table[j][ny - i - 1];

    file_stream.close();

    return GridData(coordinate_values, table, missing);
  }


  /**
   * Parse the grid size and resolution information contained in the file
   *     <filename>.geodat
   * This is a helper function for readGeoDat (below).
   */
  static void readGeoDatInfo(
    const std::string& filename,
    unsigned int& nx,
    unsigned int& ny,
    double& dx,
    double& dy,
    double& xo,
    double& yo
  )
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
  {
    unsigned int nx, ny;
    double dx, dy, xo, yo;
    readGeoDatInfo(filename, nx, ny, dx, dy, xo, yo);

    std::ifstream geodat_file(filename, std::ios::in | std::ios::binary);

    float q;
    unsigned char temp[sizeof(float)];
    std::vector<float> vals;

    while (not geodat_file.eof()) {
      // If a float isn't 4 bytes on your system, then may God have
      // mercy on your soul.
      geodat_file.read(reinterpret_cast<char*>(temp), sizeof(float));
      q = ntohx(reinterpret_cast<float&>(temp));
      vals.push_back(q);
    }

    geodat_file.close();

    std::vector<double> x(nx);
    std::vector<double> y(ny);
    Table<2, double> table(nx, ny);

    for (unsigned int i = 0; i < ny; ++i) y[i] = yo + i * dy;
    for (unsigned int j = 0; j < nx; ++j) x[j] = xo + j * dx;

    std::array<std::vector<double>, 2> coordinate_values = {{x, y}};

    for (unsigned int i = 0; i < ny; ++i)
      for (unsigned int j = 0; j < nx; ++j)
        table[j][i] = vals[ny * j + i];

    const double missing = -2.0e9;
    return GridData(coordinate_values, table, missing);
  }


  GridData readGeoTIFF(const std::string& filename)
  {
    GDALAllRegister();
    GDALDataset *data = (GDALDataset *) GDALOpen(filename.c_str(), GA_ReadOnly);
    if (data == 0) throw;

    unsigned int nx = data->GetRasterXSize(),
      ny = data->GetRasterYSize();

    double geoTransform[6];
    data->GetGeoTransform(geoTransform);
    double dx = geoTransform[1];
    double dy = geoTransform[5];
    double xo = geoTransform[0];
    double yo = geoTransform[3];

    GDALRasterBand *band = data->GetRasterBand(1);
    const double missing = band->GetNoDataValue();

    // Bro, do you even know how to C++?
    double *scanline = (double *) CPLMalloc(sizeof(double) * nx);

    Table<2, double> table(nx, ny);

    for (unsigned int i = 0; i < ny; ++i) {
      band->RasterIO( GF_Read, 0, i, nx, 1, scanline, nx, 1,
                      GDT_Float64, 0, 0 );
      for (unsigned int j = 0; j < nx; ++j)
        table[j][ny - i - 1] = scanline[j];
    }

    CPLFree(scanline);
    delete data;

    std::vector<double> x(nx);
    std::vector<double> y(ny);

    for (unsigned int i = 0; i < ny; ++i) y[i] = yo + (ny - i - 1) * dy;
    for (unsigned int j = 0; j < nx; ++j) x[j] = xo + j * dx;

    std::array<std::vector<double>, 2> coordinate_values = {{x, y}};

    return GridData(coordinate_values, table, missing);
  }

}
