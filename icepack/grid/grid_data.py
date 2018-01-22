
import numpy.ma as ma

class GridData(object):
    """Class for data sets defined on a regular spatial grid"""

    def __init__(self, origin, grid_spacing, data,
                 *, mask=None, missing_data_value=None):
        """Create a new gridded data set

        There are several ways to specify the missing data mask:
        * pass in a numpy masked array for the `data` argument
        * pass in the array `mask` of boolean values to indicate where data
          is missing
        * pass in a specific value `missing_data_value` indicating that any
          entry in `data` with this exact value is actually a missing data
          point

        Parameters
        ----------
        origin : tuple of float
            coordinates of the lower-left point of the gridded data
        grid_spacing : float
            the width of a grid cell in either coordinate direction
        data : np.ndarray or ma.MaskedArray
            values of the gridded data set
        mask : ndarray of bool, optional
            array describing missing data values; `True` indicates missing
        missing_data_value : float, optional
            value in `data` to indicate missing data
        """
        self._origin = origin
        self._delta = grid_spacing

        if isinstance(data, ma.MaskedArray):
            self.data = data
        elif missing_data_value is not None:
            self.data = ma.masked_equal(data, missing_data_value)
        elif mask is not None:
            self.data = ma.MaskedArray(data=data, mask=mask)
        else:
            raise TypeError()

    def __getitem__(self, indices):
        """Retrieve a given entry from the raw data"""
        i, j = indices
        return self.data[i, j]

    @property
    def shape(self):
        return self.data.shape

    def coordinate(self, i, j):
        """Return the coordinates of a given grid cell"""
        ny, nx = self.shape
        if not ((0 <= i < ny) and (0 <= j < nx)):
            raise IndexError()

        x0, delta = self._origin, self._delta
        return (x0[0] + j * delta, x0[1] + i * delta)

    def _index_of_point(self, x):
        """Return the index of the grid point to the lower left of a point"""
        ny, nx = self.shape
        x0, x1 = self.coordinate(0, 0), self.coordinate(ny - 1, nx - 1)

        if not ((x0[0] <= x[0] <= x1[0]) and (x0[1] <= x[1] <= x1[1])):
            raise ValueError("{0} not contained in gridded data".format(x))

        i = int((x[1] - x0[1]) / self._delta)
        j = int((x[0] - x0[0]) / self._delta)
        return min(i, ny - 2), min(j, nx - 2)

    def _is_missing(self, i, j):
        """Returns `True` if there is data missing around an index"""
        return any([self.data[k, l] is ma.masked
                    for k in (i, i + 1) for l in (j, j + 1)])

    def is_masked(self, x):
        """Returns `True` if the data cannot be interpolated to a point"""
        i, j = self._index_of_point(x)
        return self._is_missing(i, j)

    def subset(self, xmin, xmax):
        """Return a sub-sample for the region between two points"""
        ny, nx = self.shape
        x0, x1 = self.coordinate(0, 0), self.coordinate(ny - 1, nx - 1)

        Xmin = (max(xmin[0], x0[0]), max(xmin[1], x0[1]))
        Xmax = (min(xmax[0], x1[0]), min(xmax[1], x1[1]))

        imin, jmin = self._index_of_point(Xmin)
        imax, jmax = self._index_of_point(Xmax)

        data = self.data[imin: imax + 2, jmin: jmax + 2]
        return GridData(self.coordinate(imin, jmin), self._delta, data)

    def __call__(self, x):
        """Evaluate the gridded data set at a given point"""
        i, j = self._index_of_point(x)
        if self._is_missing(i, j):
            raise ValueError("Not enough data to interpolate value at {0}"
                             .format(x))
        x0 = self.coordinate(i, j)

        delta = self._delta
        a = (x[0] - x0[0]) / delta, (x[1] - x0[1]) / delta

        data = self.data
        dq_dx = data[i, j + 1] - data[i, j]
        dq_dy = data[i + 1, j] - data[i, j]
        d2q_dx_dy = data[i, j] + data[i+1, j+1] - data[i, j+1] - data[i+1, j]

        return data[i, j] + a[0]*dq_dx + a[1]*dq_dy + a[0]*a[1]*d2q_dx_dy

