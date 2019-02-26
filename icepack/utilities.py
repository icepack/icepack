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

import inspect
import numpy as np
import firedrake

def diameter(mesh):
    r"""Compute the diameter of the mesh in the L-infinity metric"""
    X = mesh.coordinates.dat.data_ro
    xmin = mesh.comm.allreduce(np.min(X, axis=0), op=np.minimum)
    xmax = mesh.comm.allreduce(np.max(X, axis=0), op=np.maximum)
    return np.max(xmax - xmin)


def depth_average(q3d, weight=firedrake.Constant(1)):
    r"""Return the weighted depth average of a function on an extruded mesh"""
    element3d = q3d.ufl_element()

    # Create the element `E x DG0` where `E` is the horizontal element for the
    # input field
    element_z = firedrake.FiniteElement(family='DG', cell='interval', degree=0)
    shape = q3d.ufl_shape
    if len(shape) == 0:
        element_xy = element3d.sub_elements()[0]
        element_avg = firedrake.TensorProductElement(element_xy, element_z)
        element2d = element_xy
    elif len(shape) == 1:
        element_xy = element3d.sub_elements()[0].sub_elements()[0]
        element_u = firedrake.TensorProductElement(element_xy, element_z)
        element_avg = firedrake.VectorElement(element_u, dim=shape[0])
        element2d = firedrake.VectorElement(element_xy, dim=shape[0])
    else:
        raise NotImplementedError('Depth average of tensor fields not yet '
                                  'implemented!')

    # Project the weighted 3D field onto vertical DG0
    mesh3d = q3d.ufl_domain()
    Q_avg = firedrake.FunctionSpace(mesh3d, element_avg)
    q_avg = firedrake.project(weight * q3d, Q_avg)

    # Create a function space on the 2D mesh and a 2D function defined on this
    # space, then copy the raw vector of expansion coefficients from the 3D DG0
    # field into the coefficients of the 2D field. TODO: Get some assurance
    # from the firedrake folks that this will always work.
    mesh2d = mesh3d._base_mesh
    Q2D = firedrake.FunctionSpace(mesh2d, element2d)
    q2d = firedrake.Function(Q2D)
    q2d.dat.data[:] = q_avg.dat.data_ro[:]

    return q2d


def add_kwarg_wrapper(func):
    signature = inspect.signature(func)
    if any(str(signature.parameters[param].kind) == 'VAR_KEYWORD'
           for param in signature.parameters):
        return func

    params = signature.parameters

    def wrapper(*args, **kwargs):
        kwargs_ = dict((key, kwargs[key]) for key in kwargs if key in params)
        return func(*args, **kwargs_)

    return wrapper
