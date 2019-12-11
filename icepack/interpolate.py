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

def interpolate(f, Q):
    r"""Interpolate an expression or a gridded data set to a function space

    Parameters
    ----------
    f : icepack.grid.GridData or tuple of icepack.grid.GridData
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
    num_points = X.shape[0]

    q = firedrake.Function(Q)

    if isinstance(f, rasterio.DatasetReader):
        q.dat.data[:] = np.fromiter(f.sample(X, indexes=1),
                                    dtype=np.float64, count=num_points)
    elif (isinstance(f, tuple) and
          all(isinstance(fi, rasterio.DatasetReader) for fi in f)):
        for i, fi in enumerate(f):
            q.dat.data[:, i] = np.fromiter(fi.sample(X, indexes=1),
                                           dtype=np.float64, count=num_points)
    else:
        raise ValueError('Argument must be a rasterio data set or a tuple of '
                         'data sets!')

    return q
