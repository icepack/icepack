# Copyright (C) 2017-2026 by Daniel Shapero <shapero@uw.edu>
# Andrew Hoffman <hoffmaao@uw.edu>
# Jessica Badgeley <badgeley@uw.edu>
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

r"""Miscellaneous utilities for depth-averaging 3D fields, computing
horizontal gradients of 3D fields, lifting 2D fields into 3D, etc."""

from operator import itemgetter
import inspect
import numpy as np
import firedrake
from firedrake import sqrt, tr, det
from .constants import ice_density as ρ_I, water_density as ρ_W


default_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "nleqerr",
    "ksp_type": "gmres",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def eigenvalues(a):
    r"""Return a pair of symbolic expressions for the largest and smallest
    eigenvalues of a 2D rank-2 tensor"""
    tr_a = tr(a)
    det_a = det(a)
    # TODO: Fret about numerical stability
    Δ = sqrt(tr_a**2 - 4 * det_a)
    return ((tr_a + Δ) / 2, (tr_a - Δ) / 2)


def diameter(mesh):
    r"""Compute the diameter of the mesh in the L-infinity metric"""
    X = mesh.coordinates.dat.data_ro
    xmin = mesh.comm.allreduce(np.min(X, axis=0), op=np.minimum)
    xmax = mesh.comm.allreduce(np.max(X, axis=0), op=np.maximum)
    return np.max(xmax - xmin)


def compute_surface(**kwargs):
    r"""Return the ice surface elevation consistent with a given
    thickness and bathymetry

    If the bathymetry beneath a tidewater glacier is too low, the ice
    will go afloat. The surface elevation of a floating ice shelf is

    .. math::
       s = (1 - \rho_I / \rho_W)h,

    provided everything is in hydrostatic balance.
    """
    # TODO: Remove the 'h' and 'b' arguments once these are deprecated.
    h = kwargs.get("thickness", kwargs.get("h"))
    b = kwargs.get("bed", kwargs.get("b"))

    Q = h.ufl_function_space()
    s_expr = firedrake.max_value(h + b, (1 - ρ_I / ρ_W) * h)
    return firedrake.Function(Q).interpolate(s_expr)


def add_kwarg_wrapper(func):
    signature = inspect.signature(func)
    if any(
        str(signature.parameters[param].kind) == "VAR_KEYWORD"
        for param in signature.parameters
    ):
        return func

    params = signature.parameters

    def wrapper(*args, **kwargs):
        kwargs_ = dict((key, kwargs[key]) for key in kwargs if key in params)
        return func(*args, **kwargs_)

    return wrapper


def lift3d(*args, **kwargs):
    raise ImportError(
        "This function has moved. If you're getting this message, it's because "
        "you're calling `icepack.utilities.lift3d`, but you want to be using "
        "`icepack.lift3d` instead."
    )


def depth_average(*args, **kwargs):
    raise ImportError(
        "This function has moved. If you're getting this message, it's because "
        "you're calling `icepack.utilities.depth_average`, but you want to be "
        "using `icepack.depth_average` instead."
    )


def vertical_velocity(*args, **kwargs):
    raise ImportError(
        "This function has moved. If you're getting this message, it's because "
        "you're calling `icepack.utilities.vertical_velocity`, but you want to be "
        "using `icepack.vertical_velocity` instead."
    )
