
import sys
import math
import numpy as np
from scipy import spatial

from icepack.grid import arcinfo, GridData
from icepack.mesh import gmsh, QuadMesh


def _interpolate(q, i, j, m):
    weight = 0.0
    Q = 0.0
    for k in range(i - m + 1, i + m):
        for l in range(j - m + 1, j + m):
            if q.data[k, l] != q.missing:
                r = np.sqrt((k - i)**2 + (l - j)**2)
                w = 1.0 / (1 + r**3)
                weight += w
                Q += w * q.data[k, l]

    if weight != 0:
        return Q / weight

    return q.missing


def fill_missing_data_points(q, mesh, radius = 7.5e3):
    X = np.array([mesh.x, mesh.y]).transpose()
    kdt = spatial.KDTree(X)

    p = np.copy(q.data)

    dx = q.x[1] - q.x[0]
    m = int(math.ceil(radius/dx))
    I, J = np.nonzero(q.data == q.missing)
    for i, j in zip(I, J):
        if kdt.query_ball_point([q.x[j], q.y[i]], radius):
            p[i, j] = _interpolate(q, i, j, m)

    return GridData(q.x, q.y, p, q.missing)


if __name__ == "__main__":
    mesh_filename = sys.argv[1]
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]

    mesh = gmsh.read_quad(mesh_filename)

    q = arcinfo.read(input_filename)
    Q = fill_missing_data_points(q, mesh)
    arcinfo.write(output_filename, Q)
