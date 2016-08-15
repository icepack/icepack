
import numpy as np

def index_of_point(x, y, X, Y):
    i = int((Y - y[0]) / (y[1] - y[0]))
    j = int((X - x[0]) / (x[1] - x[0]))

    if (0 <= i < len(y) - 1 and 0 <= j < len(x) - 1):
        return i, j

    return -1, -1


def is_missing(m, q, i, j):
    return m in [q[i, j], q[i+1, j], q[i, j+1], q[i+1, j+1]]


def bilinear_interp(x, y, q, missing, X, Y):
    i, j = index_of_point(x, y, X, Y)

    if i == -1 or j == -1:
        return missing

    if is_missing(missing, q, i, j):
        return missing

    ax = (X - x[j]) / (x[1] - x[0])
    ay = (Y - y[i]) / (y[1] - y[0])

    dx_q = q[i, j+1] - q[i, j]
    dy_q = q[i+1, j] - q[i, j]
    dx_dy_q = q[i, j] + q[i+1, j+1] - q[i+1, j] - q[i, j+1]

    return q[i, j] + ax * dx_q + ay * dy_q + ax * ay * dx_dy_q



def regrid(x, y, q, missing, X, Y):
    """
    Interpolate a gridded data set `x, y, q` to a new grid `X, Y`
    """

    nX, nY = len(X), len(Y)
    dx, dy = x[1] - x[0], y[1] - y[0]

    Q = missing * np.ones((nY, nX), dtype = np.float64)

    for I in range(nY):
        for J in range(nX):
            Q[I, J] = bilinear_interp(x, y, q, missing, X[J], Y[I])

    return Q
