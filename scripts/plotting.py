
import numpy as np
import matplotlib.path as mplPath


#
def cell_to_path(x, y, cell, k):
    p = []
    for i in range(4):
        p.append([x[cell[k,i]], y[cell[k,i]]])

    return mplPath.Path(np.array([p[0], p[1], p[2], p[3]]))


# -------------------------------------------
def cell_containing_point(x, y, cell, x0, y0):
    """
    Given a quad mesh and a point, return the cell containing that point.
    """

    # NB: this takes O(num_cells) time
    # Not optimized, quick + dirty implementation for debugging
    num_cells, _ = np.shape(cell)
    for n in range(num_cells):
        quad = cell_to_path(x, y, cell, n)
        if quad.contains_point((x0, y0)):
            return n

    return -1


# ------------------------------------
def interpolate(x, y, cell, q, x0, y0):
    """
    Given a quad mesh, nodal data and a point, interpolate the nodal data to
    the given point.
    """
    k = cell_containing_point(x, y, cell, x0, y0)

    dx = x[cell[k, 1]] - x[cell[k, 0]]
    dy = y[cell[k, 3]] - y[cell[k, 0]]

    px = (x0 - x[cell[k, 0]]) / dx
    py = (y0 - y[cell[k, 0]]) / dy

    return (q[cell[k, 0]] +
            px * (q[cell[k, 1]] - q[cell[k, 0]]) +
            py * (q[cell[k, 3]] - q[cell[k, 0]]) +
            px * py * (q[cell[k, 0]] + q[cell[k, 2]] - q[cell[k, 1]] - q[cell[k, 3]]))


# ---------------------------------
def transect(x, y, cell, q, xt, yt):
    """
    Given the UCD mesh in x, y, cell the nodal values q of some field, and
    a line passing through the mesh, return the values of the data along
    that transect.
    """

    nt = len(xt)
    qt = np.zeros(nt, dtype = np.float64)
    for i in range(nt):
        qt[i] = interpolate(x, y, cell, q, xt[i], yt[i])

    return qt
