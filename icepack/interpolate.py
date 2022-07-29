# Copyright (C) 2017-2022 by Daniel Shapero <shapero@uw.edu> and David
# Lilien
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

from functools import singledispatch
from collections.abc import Sequence
import numpy as np
import ufl
import firedrake
import rasterio
import xarray
from scipy.interpolate import RegularGridInterpolator


@singledispatch
def _sample(dataset, X, **kwargs):
    raise TypeError(
        "Input must be a single or sequence of `rasterio.DatasetReader` or "
        "`xarray.DataArray`!"
    )


@_sample.register
def _sample_rasterio_scalar(dataset: rasterio.DatasetReader, X, **kwargs):
    xres = dataset.res[0]
    yres = dataset.res[1]
    bounds = dataset.bounds
    xmin = max(X[:, 0].min() - 3 * xres, bounds.left)
    xmax = min(X[:, 0].max() + 3 * xres, bounds.right)
    ymin = max(X[:, 1].min() - 3 * yres, bounds.bottom)
    ymax = min(X[:, 1].max() + 3 * yres, bounds.top)

    window = rasterio.windows.from_bounds(
        left=xmin,
        right=xmax,
        bottom=ymin,
        top=ymax,
        transform=dataset.transform,
    )
    window = window.round_lengths(op="ceil").round_offsets(op="floor")
    transform = rasterio.windows.transform(window, dataset.transform)

    upper_left = transform * (0, 0)
    lower_right = transform * (window.width - 1, window.height - 1)
    xs = np.linspace(upper_left[0], lower_right[0], window.width)
    ys = np.linspace(lower_right[1], upper_left[1], window.height)

    data = np.flipud(dataset.read(indexes=1, window=window, masked=True)).T
    method = kwargs.get("method", "linear")
    interpolator = RegularGridInterpolator((xs, ys), data, method=method)
    return interpolator(X, method=method)


@_sample.register
def _xarray_sample(dataset: xarray.DataArray, X, **kwargs):
    x = xarray.DataArray(X[:, 0], dims="z")
    y = xarray.DataArray(X[:, 1], dims="z")
    method = kwargs.get("method", "linear")
    return dataset.interp(x=x, y=y, method=method).to_numpy()


@_sample.register
def _sample_vector(f: Sequence, X, **kwargs):
    return np.column_stack([_sample(fi, X, **kwargs) for fi in f])


def interpolate(f, Q, **kwargs):
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

    # Cannot take sub-elements if function is 3D scalar, otherwise shape will
    # mismatch vertical basis. This attempts to distinguish if multiple
    # subelements due to dimension or vector function.
    if issubclass(type(element), firedrake.VectorElement):
        element = element.sub_elements()[0]

    V = firedrake.VectorFunctionSpace(mesh, element)
    X = firedrake.interpolate(mesh.coordinates, V).dat.data_ro[:, :2]

    q = firedrake.Function(Q)
    q.dat.data[:] = _sample(f, X, **kwargs)
    return q
