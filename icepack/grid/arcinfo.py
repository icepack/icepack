"""Functions for reading/writing the ArcInfo ASCII format

This module contains functions for reading and writing raster data in the
`Arc/Info ASCII grid <https://en.wikipedia.org/wiki/Esri_grid>`_ format.
The Center for Remote Sensing of Ice Sheets releases their `gridded data
products <https://data.cresis.ku.edu/data/grids/>`_ in this format. The
Arc/Info ASCII format has the virtue of being human readable, easy to
parse, and readable by every geographic information system.
"""
import numpy as np
from icepack.grid import GridData

def write(file, q, missing):
    """Write a gridded data set to an ArcInfo ASCII format file

    Parameters
    ----------
    filename
       the destination file
    q : icepack.grid.GridData
       the gridded data set to be written
    """
    nx, ny = len(q.x), len(q.y)

    file.write("ncols           {0}\n".format(nx))
    file.write("nrows           {0}\n".format(ny))
    file.write("xllcorner       {0}\n".format(q.x[0]))
    file.write("yllcorner       {0}\n".format(q.y[0]))
    file.write("cellsize        {0}\n".format(q.x[1] - q.x[0]))
    file.write("NODATA_value    {0}\n".format(missing))

    for i in range(ny - 1, -1, -1):
        for j in range(nx):
            value = q[i, j] if not q.data.mask[i, j] else missing
            file.write("{0} ".format(value))
        file.write("\n")


def read(file):
    """Read an ArcInfo ASCII file into a gridded data set

    Parameters
    ----------
    file
        the source file

    Returns
    -------
    q : icepack.grid.GridData
        the gridded data set's coordinates, values, and missing data mask
    """
    def rd():
        return file.readline().split()[1]

    nx = int(rd())
    ny = int(rd())
    xo = float(rd())
    yo = float(rd())
    dx = float(rd())
    missing = float(rd())

    x = np.array([xo + dx * i for i in range(nx)], dtype = np.float64)
    y = np.array([yo + dx * i for i in range(ny)], dtype = np.float64)
    data = np.zeros((ny, nx), dtype = np.float64)

    for i in range(ny-1, -1, -1):
        data[i, :] = [float(q) for q in file.readline().split()]

    return GridData(x, y, data, missing)

