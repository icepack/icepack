
import numpy as np

from icepack.grid import GridData

# --------------------
def write(filename, q):
    """
    Write a gridded data set to an ArcInfo ASCII format file, which can be read
    by GIS software such as ArcGIS or QGIS.

    Parameters:
    ==========
    filename: the destination file
    q: a GridData object
    """
    nx = len(q.x)
    ny = len(q.y)

    with open(filename, "w") as fid:
        fid.write("ncols           {0}\n".format(nx))
        fid.write("nrows           {0}\n".format(ny))
        fid.write("xllcorner       {0}\n".format(q.x[0]))
        fid.write("yllcorner       {0}\n".format(q.y[0]))
        fid.write("cellsize        {0}\n".format(q.x[1] - q.x[0]))
        fid.write("NODATA_value    {0}\n".format(q.missing))

        for i in range(ny-1, -1, -1):
            for j in range(nx):
                fid.write("{0} ".format(q.data[i, j]))
            fid.write("\n")


# ----------------
def read(filename):
    """
    Read an ArcInfo ASCII file into a gridded data set.

    Returns:
    =======
    x, y: coordinates of the grid points
    data: data values at the grid points
    missing: value to indicate missing data at a grid point
    """
    with open(filename, "r") as fid:
        def rd():
            return fid.readline().split()[1]

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
            data[i, :] = [float(q) for q in fid.readline().split()]

        return GridData(x, y, data, missing)
