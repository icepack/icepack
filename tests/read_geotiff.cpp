

#include <gdal/gdal_priv.h>
#include <gdal/cpl_conv.h>
#include <gdal/cpl_string.h>
#include <gdal/ogr_spatialref.h>

#include "read_gridded_data.hpp"

using dealii::Point;


bool generateExampleGeoTIFF(const std::string& filename)
{
  double x0 = 0.0, y0 = 0.0;
  double dx = 2, dy = -2;
  constexpr unsigned int nx = 301, ny = 201;

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
  double geoTransform[6] = { x0, dx, 0, y0, 0, dy };

  OGRSpatialReference oSRS;
  char *SRS_WKT = 0;
  GDALRasterBand *band;
  double raster[nx * ny];

  double x, y;
  for (unsigned int i = 0; i < ny; ++i)
  {
    y = y0 + i * dy;
    for (unsigned int j = 0; j < nx; ++j)
    {
      x = x0 + j * dx;
      raster[nx * i + j] = 1 + x * y;
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
  GridData q = readGeoTiff(filename);

  std::cout << q.xmin() << ", " << q.xmax() << std::endl;
  std::cout << q.ymin() << ", " << q.ymax() << std::endl;

  std::cout << q.value(Point<2> {444720, 3751320}) << std::endl;
  std::cout << q.value(Point<2> {444750, 3751290}) << std::endl;

  return 0;
}
