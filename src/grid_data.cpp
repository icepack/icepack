
#include <fstream>
#include <sstream>
#include <vector>

#include <gdal_priv.h>
#include <cpl_conv.h>

#include <deal.II/base/table.h>

#include <icepack/grid_data.hpp>

using dealii::Table;
using dealii::TableIndices;
using dealii::Functions::InterpolatedTensorProductGridData;

namespace icepack {

  namespace {
    Table<2, bool>
    make_missing_data_mask(const Table<2, double>& data, const double missing)
    {
      Table<2, bool> mask(data.size(0), data.size(1));

      for (size_t i = 0; i < data.size(0); ++i)
        for (size_t j = 0; j < data.size(1); ++j)
          mask(i, j) = (data(i, j) == missing);

      return mask;
    }
  }


  /* -------------------
   * Methods of GridData
   * ------------------- */

  GridData::GridData(
    const std::array<std::vector<double>, 2>& coordinate_values,
    const Table<2, double>& data_values,
    const double missing
  ) :
    InterpolatedTensorProductGridData<2>(coordinate_values, data_values),
    xrange{{coordinate_values[0][0], coordinate_values[0].back()}},
    yrange{{coordinate_values[1][0], coordinate_values[1].back()}},
    missing(missing),
    mask(make_missing_data_mask(data_values, missing))
  {}


  double GridData::value(const Point<2>& x, const unsigned int) const
  {
    if (is_masked(x))
      return missing;

    return InterpolatedTensorProductGridData<2>::value(x);
  }


  bool GridData::is_masked(const Point<2>& x) const
  {
    const auto idx = table_index_of_point(x);
    return
      mask(idx[0], idx[1]) || mask(idx[0] + 1, idx[1]) ||
      mask(idx[0], idx[1] + 1) || mask(idx[0] + 1, idx[1] + 1);
  }


  /* ---------------------------------------------------
   * Procedures for reading various gridded data formats
   * --------------------------------------------------- */

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
    double * line = (double *) CPLMalloc(sizeof(double) * nx);

    Table<2, double> table(nx, ny);

    for (unsigned int i = 0; i < ny; ++i) {
      CPLErr cpl_err =
        band->RasterIO(GF_Read, 0, i, nx, 1, line, nx, 1, GDT_Float64, 0, 0);
      for (unsigned int j = 0; j < nx; ++j)
        table[j][ny - i - 1] = line[j];
    }

    CPLFree(line);
    delete data;

    std::vector<double> x(nx);
    std::vector<double> y(ny);

    for (unsigned int i = 0; i < ny; ++i) y[i] = yo + (ny - i - 1) * dy;
    for (unsigned int j = 0; j < nx; ++j) x[j] = xo + j * dx;

    std::array<std::vector<double>, 2> coordinate_values = {{x, y}};

    return GridData(coordinate_values, table, missing);
  }

}
