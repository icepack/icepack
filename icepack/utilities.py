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


def diameter(mesh):
    """Compute the diameter of the mesh in the L-infinity metric"""
    X = mesh.coordinates.dat.data
    _, d = np.shape(X)
    Ls = np.array([np.max(X[:, k]) - np.min(X[:, k]) for k in range(d)])
    return np.min(Ls)


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

