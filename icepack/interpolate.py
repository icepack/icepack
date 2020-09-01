# Copyright (C) 2017-2019 by Daniel Shapero <shapero@uw.edu>
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

r"""Functions for interpolating gridded remote sensing data sets to finite
element spaces"""

import numpy as np
import ufl
import firedrake
import rasterio
from scipy.interpolate import RegularGridInterpolator


def _sample(dataset, X, method):
    xres = dataset.res[0]
    bounds = dataset.bounds
    xmin = max(X[:, 0].min() - 2 * xres, bounds.left)
    xmax = min(X[:, 0].max() + 2 * xres, bounds.right)
    ymin = max(X[:, 1].min() - 2 * xres, bounds.bottom)
    ymax = min(X[:, 1].max() + 2 * xres, bounds.top)

    window = rasterio.windows.from_bounds(
        left=xmin,
        right=xmax,
        bottom=ymin,
        top=ymax,
        width=dataset.width,
        height=dataset.height,
        transform=dataset.transform
    )
    window = window.round_lengths(op='ceil').round_offsets(op='floor')
    transform = rasterio.windows.transform(window, dataset.transform)

    upper_left = transform * (0, 0)
    lower_right = transform * (window.width - 1, window.height - 1)
    xs = np.linspace(upper_left[0], lower_right[0], window.width)
    ys = np.linspace(lower_right[1], upper_left[1], window.height)

    data = np.flipud(dataset.read(indexes=1, window=window, masked=True)).T
    interpolator = RegularGridInterpolator((xs, ys), data, method=method)
    return interpolator(X, method=method)


def interpolate(f, Q, method='linear'):
    r"""Interpolate an expression or a gridded data set to a function space

    Parameters
    ----------
    f : rasterio dataset or tuple of rasterio datasets
        The gridded data set for scalar fields or the tuple of gridded data
        sets for each component
    Q : firedrake.FunctionSpace
        The function space where the result will live

    Returns
    -------
    firedrake.Function
        A finite element function defined on `Q` with the same nodal values
        as the data `f`
    """
    if isinstance(f, (ufl.core.expr.Expr, firedrake.Function)):
        return firedrake.interpolate(f, Q)

    mesh = Q.mesh()
    element = Q.ufl_element()
    if len(element.sub_elements()) > 0:
        element = element.sub_elements()[0]

    V = firedrake.VectorFunctionSpace(mesh, element)
    X = firedrake.interpolate(mesh.coordinates, V).dat.data_ro

    q = firedrake.Function(Q)

    if isinstance(f, rasterio.DatasetReader):
        q.dat.data[:] = _sample(f, X, method)
    elif (isinstance(f, tuple) and
          all(isinstance(fi, rasterio.DatasetReader) for fi in f)):
        for i, fi in enumerate(f):
            q.dat.data[:, i] = _sample(fi, X, method)
    else:
        raise ValueError('Argument must be a rasterio data set or a tuple of '
                         'data sets!')

    return q
