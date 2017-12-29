
import numpy.ma as ma

def _index_of_point(x, y, X, Y):
    i = int((Y - y[0]) / (y[1] - y[0]))
    j = int((X - x[0]) / (x[1] - x[0]))

    if (0 <= i < len(y) - 1) and (0 <= j < len(x) - 1):
        return i, j

    raise ValueError('Point ({0}, {1}) not contained in the gridded data'
                     .format(X, Y))


def _is_missing(q, i, j):
    return any([q[k, l] is ma.masked for k in (i, i+1) for l in (j, j+1)])


def _bilinear_interp(q, X, Y):
    x, y = q.x, q.y
    i, j = _index_of_point(x, y, X, Y)
    if _is_missing(q.data, i, j):
        raise ValueError('Not enough data to interpolate value at ({0}, {1})'
                         .format(X, Y))

    ax, ay = (X - x[j])/(x[1] - x[0]), (Y - y[i])/(y[1] - y[0])
    dq_dx = q[i, j+1] - q[i, j]
    dq_dy = q[i+1, j] - q[i, j]
    d2q_dx_dy = q[i, j] + q[i+1, j+1] - q[i+1, j] - q[i, j+1]

    return q[i, j] + ax*dq_dx + ay*dq_dy + ax*ay*d2q_dx_dy


class GridData(object):
    def __init__(self, x, y, data, missing_data_value):
        ny, nx = data.shape
        if (len(x) != nx) or (len(y) != ny):
            raise ValueError('Incompatible input array sizes')

        self.x = x
        self.y = y
        self.data = ma.masked_equal(data, missing_data_value)

    def __getitem__(self, indices):
        i, j = indices
        return self.data[i, j]

    def is_masked(self, x):
        i, j = _index_of_point(self.x, self.y, x[0], x[1])
        return _is_missing(self.data, i, j)

    def __call__(self, x):
        return _bilinear_interp(self, x[0], x[1])

