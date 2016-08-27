
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <cpl_string.h>
#include <ogr_spatialref.h>
#include <icepack/grid_data.hpp>
#include "../testing.hpp"

using namespace icepack;
using dealii::Point;


// Some global constants for the mesh size
constexpr unsigned int nx = 11, ny = 6;
constexpr double xo = 0.0, yo = 0.0;
constexpr double dx = 1.0, dy = -1.0;


bool generateExampleGeoTIFF(const std::string& filename)
{
  const char format[] = "GTiff";
  GDALDriver * driver = GetGDALDriverManager()->GetDriverByName(format);
  if (not driver) {
    std::cout << "Unable to get geotif GDAL driver!" << std::endl;
    return false;
  }

  char ** options = 0;
  GDALDataset * data =
    driver->Create(filename.c_str(), nx, ny, 1, GDT_Float64, options);

  double geoTransform[6] = { xo, dx, 0, yo - (ny - 1) * dy, 0, dy };

  OGRSpatialReference oSRS;
  char *SRS_WKT = 0;

  double raster[nx * ny];
  double x, y;
  for (unsigned int i = 0; i < ny; ++i) {
    y = yo - i * dy;
    for (unsigned int j = 0; j < nx; ++j) {
      x = xo + j * dx;
      raster[nx * (ny - i - 1) + j] = 1 + x * y;
    }
  }

  data->SetGeoTransform(geoTransform);

  oSRS.SetUTM( 11, TRUE );
  oSRS.SetWellKnownGeogCS( "NAD27" );
  oSRS.exportToWkt(&SRS_WKT);
  data->SetProjection(SRS_WKT);
  CPLFree(SRS_WKT);

  GDALRasterBand * band = data->GetRasterBand(1);
  CPLErr cpl_err =
    band->RasterIO(GF_Write, 0, 0, nx, ny, raster, nx, ny, GDT_Float64, 0, 0);
  GDALClose((GDALDatasetH) data);

  return true;
}



int main(int argc, char **argv)
{
  bool verbose = false;
  if (strcmp(argv[argc-1], "-v") == 0) verbose = true;

  const std::string& filename = "example_geotiff_file.tiff";

  GDALAllRegister();

  check(generateExampleGeoTIFF(filename));
  GridData example_data = readGeoTIFF(filename);

  double xmin = example_data.xrange[0],
         xmax = example_data.xrange[1],
         ymin = example_data.yrange[0],
         ymax = example_data.yrange[1];

  check(xmin == xo & xmax == xo + (nx - 1) * dx &
        ymin == yo & ymax == yo + (ny - 1) * fabs(dy));

  double x, y;
  double z, w;
  Point<2> p;
  for (unsigned int i = 0; i < ny; ++i) {
    y = yo + i * fabs(dy);
    for (unsigned int j = 0; j < ny; ++j) {
      x = xo + j * dx;

      p = {x, y};

      z = 1 + x * y;
      w = example_data.value(p, 0);

      check(fabs(w - z) < 1.0e-12);
    }
  }

  if (verbose) std::cout << "Reading GeoTIFF data worked!" << std::endl;
  return 0;
}
