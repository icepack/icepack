# Copyright (C) 2017-2018 by Daniel Shapero <shapero@uw.edu>
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

import ufl
import firedrake
import icepack.grid

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
    X = firedrake.interpolate(mesh.coordinates, V)

    if isinstance(f, icepack.grid.GridData):
        F = f
    elif isinstance(f, tuple):
        if all(isinstance(fi, icepack.grid.GridData) for fi in f):
            F = lambda x: tuple(fi(x) for fi in f)
    else:
        raise ValueError('Argument must be a GridData or a tuple of GridData!')

    q = firedrake.Function(Q)
    for i in range(q.dat.data_ro.shape[0]):
        q.dat.data[i] = F(X.dat.data_ro[i, :])

    return q
