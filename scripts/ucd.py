
import numpy as np


# ----------------
def read(filename):
    """
    Read in a .ucd file containing a 2D deal.II quad mesh and velocity data.
    """
    with open(filename, 'r') as f:
        num_points, num_cells, _, _, _ = map(int, f.readline().split())

        x = np.zeros(num_points, dtype = np.float64)
        y = np.zeros(num_points, dtype = np.float64)

        for i in range(num_points):
            _, x[i], y[i], _ = map(float, f.readline().split())

        f.readline()

        cell = np.zeros((num_cells, 4), dtype = int)

        for i in range(num_cells):
            cell[i, :] = map(lambda k: int(k) - 1, f.readline().split()[3:])

        for i in range(4):
            f.readline()

        u = np.zeros(num_points, dtype = np.float64)
        v = np.zeros(num_points, dtype = np.float64)

        for i in range(num_points):
            _, u[i], v[i] = map(float, f.readline().split())

    return x, y, cell, u, v
