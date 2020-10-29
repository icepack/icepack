# Copyright (C) 2017-2020 by Daniel Shapero <shapero@uw.edu>
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

import firedrake
from firedrake import dx, inner
from icepack import utilities


class Continuity:
    r"""Describes the form of the mass continuity equation"""
    def __init__(self, dimension):
        if dimension == 2:
            self.facet_normal = firedrake.FacetNormal
            self.grad = firedrake.grad
            self.div = firedrake.div
            self.ds = firedrake.ds
        elif dimension == 3:
            self.facet_normal = utilities.facet_normal_2
            self.grad = utilities.grad_2
            self.div = utilities.div_2
            self.ds = firedrake.ds_v
        else:
            raise ValueError('Dimension must be 2 or 3!')

    def __call__(self, dt, **kwargs):
        keys = ('thickness', 'velocity', 'accumulation')
        keys_alt = ('h', 'u', 'a')
        h, u, a = utilities.get_kwargs_alt(kwargs, keys, keys_alt)
        h_inflow = kwargs.get('thickness_inflow', kwargs.get('h_inflow', h))

        Q = h.function_space()
        q = firedrake.TestFunction(Q)

        grad, ds, n = self.grad, self.ds, self.facet_normal(Q.mesh())
        u_n = inner(u, n)
        flux_cells = -inner(h * u, grad(q)) * dx
        flux_out = h * firedrake.max_value(u_n, 0) * q * ds
        flux_in = h_inflow * firedrake.min_value(u_n, 0) * q * ds
        accumulation = a * q * dx
        return accumulation - (flux_in + flux_out + flux_cells)
