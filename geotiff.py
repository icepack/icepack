#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# David Lilien <dlilien90@gmail.com>
#

"""
Utility for reading geotifs into the GridData class

depends on gdal
"""

import numpy as np
from icepack.grid import GridData
from osgeo import gdal
driver = gdal.GetDriverByName('GTiff')


def write(fn, q, missing=-9999, t_srs=None):
    """Write a gridded dataset to a GeoTIFF file

    Parameters
    ----------
    fn: str
        Destination filename
    q: icepack.grid.GridData
        GridData to write
    missing: float, optional
        No data value (default -9999)
    """
    ny, nx = q.shape
    gridsize = q._delta
    ulx, uly = q._origin[0], q._origin[1] + gridsize * ny
    gt = (ulx, gridsize, 0, uly, 0, -gridsize)
    dst_ds = driver.Create(fn, nx, ny, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(gt)
    values = q.data.copy()
    values[q.data.mask] = missing
    if t_srs is not None:
        dst_ds.SetProjection(t_srs)

    dst_ds.GetRasterBand(1).WriteArray(np.flipud(values))
    dst_ds.GetRasterBand(1).SetNoDataValue(float(missing))


def read(fn):
    """Read a gridded geotiff

    Parameters
    ----------

    fn: str
        File to read. This does not actually need to be a geotiff, but any gdal-readable file
    """

    ds = gdal.Open(fn, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    data = np.array(band.ReadAsArray(), dtype=np.float64)
    data = np.ma.array(data)
    data.mask = np.isnan(data)
    return GridData((gt[0], gt[3] + data.shape[0] * gt[5]), gt[1], np.flipud(data), missing_data_value=None)
