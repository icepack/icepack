
import numpy as np


# ------------------------------
def _read2(ucd_file, num_points):
    ucd_file.readline()
    ucd_file.readline()

    u = np.zeros(num_points, dtype = np.float64)
    v = np.zeros(num_points, dtype = np.float64)

    for i in range(num_points):
        _, u[i], v[i] = [float(s) for s in ucd_file.readline().split()]

    return u, v


# ------------------------------
def _read1(ucd_file, num_points):
    ucd_file.readline()

    q = np.zeros(num_points, dtype = np.float64)

    for i in range(num_points):
        q[i] = float(ucd_file.readline().split()[1])

    return q


# ----------------
def read(filename):
    """
    Read in a .ucd file containing a 2D deal.II quad mesh and velocity data.
    """
    with open(filename, 'r') as f:
        num_points, num_cells, d, _, _ = [int(s) for s in f.readline().split()]

        x = np.zeros(num_points, dtype = np.float64)
        y = np.zeros(num_points, dtype = np.float64)

        for i in range(num_points):
            _, x[i], y[i], _ = [float(s) for s in f.readline().split()]

        f.readline()

        cell = np.zeros((num_cells, 4), dtype = int)

        for i in range(num_cells):
            cell[i, :] = [int(s) - 1 for s in f.readline().split()[3:]]

        # Skip over two lines
        f.readline()
        f.readline()

        q = []
        if d == 1:
            q = _read1(f, num_points)
            return x, y, cell, q

        assert(d == 2)

        u, v = _read2(f, num_points)

    return x, y, cell, u, v



# -----------------------------------
def _fixup_triangles(x, y, triangles):
    num_triangles, _ = np.shape(triangles)
    for n in range(num_triangles):
        i, j, k = triangles[n, :]
        A = np.array([[1.0, x[i], y[i]],
                      [1.0, x[j], y[j]],
                      [1.0, x[k], y[k]]])
        if np.linalg.det(A) < 0:
            triangles[n, 0] = j
            triangles[n, 1] = i

    return triangles


# -----------------------
def _subdivide_quads(cell):
    num_cells, _ = np.shape(cell)
    triangles = np.zeros((2 * num_cells, 3), dtype = int)

    for n in range(num_cells):
        i, j, k, l = cell[n, :]
        triangles[2*n] = [i, j, k]
        triangles[2*n+1] = [k, l, i]

    return triangles


# -------------------------------------
def quad_cells_to_triangles(x, y, cell):
    return _fixup_triangles(x, y, _subdivide_quads(cell))
