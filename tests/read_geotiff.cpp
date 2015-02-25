

#include <gdal/gdal_priv.h>
#include <gdal/cpl_conv.h>
#include <gdal/cpl_string.h>
#include <gdal/ogr_spatialref.h>

#include "read_gridded_data.hpp"


int main(int argc, char **argv)
{
  const char *filename = "example_geotiff_file.tiff";

  GDALAllRegister();

  const char *format = "GTiff";
  GDALDriver *driver = GetGDALDriverManager()->GetDriverByName(format);
  if (not driver) return 1;

  char **options = NULL;
  GDALDataset *data = driver->Create(filename, 512, 512, 1, GDT_Byte,
                                     options);
  double geoTransform[6] = { 444720, 30, 0, 3751320, 0, -30 };
  OGRSpatialReference oSRS;
  char *SRS_WKT = NULL;
  GDALRasterBand *band;
  GByte raster[512 * 512];

  data->SetGeoTransform(geoTransform);

  oSRS.SetUTM( 11, TRUE );
  oSRS.SetWellKnownGeogCS( "NAD27" );
  oSRS.exportToWkt(&SRS_WKT);
  data->SetProjection(SRS_WKT);
  CPLFree(SRS_WKT);

  band = data->GetRasterBand(1);
  band->RasterIO( GF_Write, 0, 0, 512, 512, raster, 512, 512, GDT_Byte, 0, 0);
  GDALClose((GDALDatasetH) data);

  //GridData q = readGeoTiff(argv[1]);

  return 0;
}
