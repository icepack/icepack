
#include <gdal/gdal_priv.h>
#include <gdal/cpl_conv.h>
#include <gdal/cpl_string.h>
#include <gdal/ogr_spatialref.h>

#include "read_gridded_data.hpp"

using dealii::Point;


// Some global constants for the mesh size
constexpr unsigned int nx = 11, ny = 6;
constexpr double xo = 0.0, yo = 0.0;
constexpr double dx = 1.0, dy = -1.0;


bool generateExampleGeoTIFF(const std::string& filename)
{
  const char format[] = "GTiff";
  GDALDriver *driver = GetGDALDriverManager()->GetDriverByName(format);
  if (not driver)
  {
    std::cout << "Unable to get geotif GDAL driver!" << std::endl;
    return false;
  }

  char **options = 0;
  GDALDataset *data = driver->Create(filename.c_str(),
                                     nx,
                                     ny,
                                     1,
                                     GDT_Float64,
                                     options);

  double geoTransform[6] = { xo, dx, 0, yo - (ny - 1) * dy, 0, dy };

  OGRSpatialReference oSRS;
  char *SRS_WKT = 0;
  GDALRasterBand *band;
  double raster[nx * ny];

  double x, y;
  for (unsigned int i = 0; i < ny; ++i)
  {
    y = yo - i * dy;
    for (unsigned int j = 0; j < nx; ++j)
    {
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

  band = data->GetRasterBand(1);
  band->RasterIO(GF_Write, 0, 0, nx, ny, raster, nx, ny, GDT_Float64, 0, 0);
  GDALClose((GDALDatasetH) data);

  return true;
}



int main(int argc, char **argv)
{
  const std::string& filename = "example_geotiff_file.tiff";

  GDALAllRegister();

  bool successfully_wrote_example = generateExampleGeoTIFF(filename);
  if (not successfully_wrote_example) return 1;
  GridData example_data = readGeoTIFF(filename);

  double xmin = example_data.xmin(),
         xmax = example_data.xmax(),
         ymin = example_data.ymin(),
         ymax = example_data.ymax();

  if (xmin != xo or xmax != xo + (nx - 1) * dx or
      ymin != yo or ymax != yo + (ny - 1) * fabs(dy))
  {
    std::cout << "Failed to record spatial extent of data." << std::endl;
    std::cout << xmin << ", " << xmax << std::endl;
    std::cout << ymin << ", " << ymax << std::endl;
    return 1;
  }

  double x, y;
  double z, w;
  Point<2> p;
  for (unsigned int i = 0; i < ny; ++i)
  {
    y = yo + i * fabs(dy);
    for (unsigned int j = 0; j < ny; ++j)
    {
      x = xo + j * dx;

      p = {x, y};

      z = 1 + x * y;
      w = example_data.value(p, 0);

      if (fabs(w - z) > 1.0e-12)
      {
        std::cout << "Reading GeoTIFF data failed." << std::endl;
        std::cout << "Correct value: " << z << std::endl;
        std::cout << "Data read:     " << w << std::endl;
        return 1;
      }
    }
  }

  std::cout << "Reading GeoTIFF data worked!" << std::endl;
  return 0;
}
