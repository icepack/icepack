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

"""Solvers for the mass continuity equation

This module contains a solver for the conservative advection equation that
describes the evolution of ice thickness. While the basic mass transport
solver will suffice for, say, ice shelf flow, other models that describe
grounded glaciers will also need to update the ice surface elevation in a
manner consistent with the bed elevation and where the ice may go afloat.
"""

import firedrake
from firedrake import grad, div, dx, ds, inner

class MassTransport(object):
    def solve(self, dt, h0, a, u, **kwargs):
        Q = h0.ufl_function_space()
        h, phi = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)
        mesh = Q.mesh()

        n = firedrake.FacetNormal(mesh)
        outflow = firedrake.max_value(inner(u, n), 0)
        inflow = firedrake.min_value(inner(u, n), 0)

        F = (h * (phi - dt * inner(u, grad(phi))) * dx
             + dt * h * phi * outflow * ds)
        A = (h0 + dt * a) * phi * dx - dt * h0 * phi * inflow * ds

        h = h0.copy(deepcopy=True)
        solver_parameters = {'ksp_type': 'preonly', 'pc_type' :'lu'}
        firedrake.solve(F == A, h, solver_parameters=solver_parameters)

        return h

