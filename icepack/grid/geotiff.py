# Copyright (C) 2018 by David Lilien <dlilien90@gmail.com>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

"""Functions for reading/writing GeoTIFF files"""

import numpy as np
from icepack.grid import GridData
import rasterio, rasterio.crs

def write(filename, q, missing=-9999, crs=None):
    """Write a gridded dataset to a GeoTIFF file

    Parameters
    ----------
    filename : str
        name of the destination file
    q : icepack.grid.GridData
        the gridded data set to be written
    missing : float, optional
        No data value (default -9999)
    crs : int, string, or rasterio.crs.CRS, optional
        Coordinate reference system, specified either as a string, an EPSG
        code, or a CRS object from rasterio
    """
    ny, nx = q.shape
    x0, y0 = q.coordinate(0, 0)
    x1 = q.coordinate(0, 1)[0]
    dx = x1 - x0
    transform = rasterio.Affine(dx, 0, x0, 0, -dx, y0 + dx * ny)

    if isinstance(crs, int):
        crs = rasterio.crs.CRS.from_epsg(crs)
    elif isinstance(crs, str):
        crs = rasterio.crs.CRS.from_string(crs)

    with rasterio.open(filename, 'w', driver='GTiff', dtype=q.data.dtype.type,
        count=1, width=nx, height=ny, transform=transform, crs=crs) as dataset:
        dataset.write(np.flipud(q.data), 1)


def read(filename):
    """Read a GeoTIFF file into a gridded data set

    Parameters
    ----------
    filename: str or file-like
        name of the input file
    """
    with rasterio.open(filename, 'r') as dataset:
        nx, ny = dataset.width, dataset.height
        x0, y0 = dataset.bounds.left, dataset.bounds.bottom

        # The `affine` attribute was renamed to `transform` in rasterio-1.0
        try:
            dx = dataset.affine[0]
        except AttributeError:
            dx = dataset.transform[0]

        data = np.flipud(dataset.read(indexes=1, masked=True))
        return GridData((x0, y0), dx, data)
