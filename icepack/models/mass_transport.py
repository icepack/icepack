# Copyright (C) 2017-2021 by Daniel Shapero <shapero@uw.edu>
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

r"""Solvers for the mass continuity equation

This module contains solvers for the conservative advection equation that
describes the evolution of ice thickness. While the basic mass transport
solver will suffice for, say, ice shelf flow, other models that describe
grounded glaciers will also need to update the ice surface elevation in a
manner consistent with the bed elevation and where the ice may go afloat.
"""

from operator import itemgetter
import firedrake
from firedrake import dx, inner
from icepack.calculus import grad, FacetNormal


class Continuity:
    r"""Describes the form of the mass continuity equation"""

    def __call__(self, dt, **kwargs):
        keys = ("thickness", "velocity", "accumulation")
        h, u, a = itemgetter(*keys)(kwargs)
        h_inflow = kwargs.get("thickness_inflow", h)

        Q = h.function_space()
        q = firedrake.TestFunction(Q)

        mesh = Q.mesh()
        n = FacetNormal(mesh)
        ds = firedrake.ds if mesh.layers is None else firedrake.ds_v

        u_n = inner(u, n)
        flux_cells = -inner(h * u, grad(q)) * dx
        flux_out = h * firedrake.max_value(u_n, 0) * q * ds
        flux_in = h_inflow * firedrake.min_value(u_n, 0) * q * ds
        accumulation = a * q * dx
        return accumulation - (flux_in + flux_out + flux_cells)
