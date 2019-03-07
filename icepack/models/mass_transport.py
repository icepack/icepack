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

r"""Solvers for the mass continuity equation

This module contains a solver for the conservative advection equation that
describes the evolution of ice thickness. While the basic mass transport
solver will suffice for, say, ice shelf flow, other models that describe
grounded glaciers will also need to update the ice surface elevation in a
manner consistent with the bed elevation and where the ice may go afloat.
"""

import firedrake
from firedrake import grad, dx, ds, inner

class MassTransport(object):
    def solve(self, dt, h0, a, u, h_inflow=None, **kwargs):
        r"""Propagate the thickness forward by one timestep

        This function uses the implicit Euler timestepping scheme to avoid
        the stability issues associated to using continuous finite elements
        for advection-type equations. The implicit Euler scheme is stable
        for any timestep; you do not need to ensure that the CFL condition
        is satisfied in order to get an answer. Nonetheless, keeping the
        timestep within the CFL bound is a good idea for accuracy.

        Parameters
        ----------
        dt : float
            Timestep
        h0 : firedrake.Function
            Initial ice thickness
        a : firedrake.Function
            Sum of accumulation and melt rates
        u : firedrake.Function
            Ice velocity
        h_inflow : firedrake.Function
            Thickness of the upstream ice that advects into the domain

        Returns
        -------
        h : firedrake.Function
            Ice thickness at `t + dt`
        """
        h_inflow = h_inflow if h_inflow is not None else h0

        Q = h0.function_space()
        h, φ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)

        n = firedrake.FacetNormal(Q.mesh())
        outflow = firedrake.max_value(inner(u, n), 0)
        inflow = firedrake.min_value(inner(u, n), 0)

        flux_cells = -h * inner(u, grad(φ)) * dx
        flux_out = h * φ * outflow * ds
        F = h * φ * dx + dt * (flux_cells + flux_out)

        accumulation = a * φ * dx
        flux_in = -h_inflow * φ * inflow * ds
        A = h0 * φ * dx + dt * (accumulation + flux_in)

        h = h0.copy(deepcopy=True)
        solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
        firedrake.solve(F == A, h, solver_parameters=solver_parameters)

        return h
