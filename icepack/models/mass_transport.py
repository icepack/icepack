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

This module contains solvers for the conservative advection equation that
describes the evolution of ice thickness. While the basic mass transport
solver will suffice for, say, ice shelf flow, other models that describe
grounded glaciers will also need to update the ice surface elevation in a
manner consistent with the bed elevation and where the ice may go afloat.
"""

import warnings
import firedrake
from firedrake import dx, inner
from icepack import utilities


class Continuity(object):
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
        keys = ('thickness', 'velocity', 'accumulation', 'thickness_inflow')
        keys_alt = ('h', 'u', 'a', 'h_inflow')
        h, u, a, h_inflow = utilities.get_kwargs_alt(kwargs, keys, keys_alt)

        Q = h.function_space()
        q = firedrake.TestFunction(Q)

        grad, ds, n = self.grad, self.ds, self.facet_normal(Q.mesh())
        u_n = inner(u, n)
        flux_cells = -inner(h * u, grad(q)) * dx
        flux_out = h * firedrake.max_value(u_n, 0) * q * ds
        flux_in = h_inflow * firedrake.min_value(u_n, 0) * q * ds
        accumulation = a * q * dx
        return accumulation - (flux_in + flux_out + flux_cells)


class MassTransport(object):
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


class ImplicitEuler(MassTransport):
    def __init__(self, dimension=2):
        super(ImplicitEuler, self).__init__(dimension)

    def solve(self, dt, h0, a, u, h_inflow=None):
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
        warnings.warn('Solving methods have moved to the FlowSolver class, '
                      'this method will be removed in future versions.',
                      FutureWarning)

        grad, ds = self.grad, self.ds

        h_inflow = h_inflow if h_inflow is not None else h0

        Q = h0.function_space()
        h, φ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)

        n = self.facet_normal(Q.mesh())
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


class LaxWendroff(MassTransport):
    def __init__(self, dimension=2):
        super(LaxWendroff, self).__init__(dimension)

    def solve(self, dt, h0, a, u, h_inflow=None):
        r"""Propagate the thickness forward by one timestep

        This function uses an implicit second-order Taylor-Galerkin (also
        known as Lax-Wendroff) scheme to solve the conservative advection
        equation for ice thickness.

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
        warnings.warn('Solving methods have moved to the FlowSolver class, '
                      'this method will be removed in future versions.',
                      FutureWarning)

        grad, div, ds = self.grad, self.div, self.ds

        h_inflow = h_inflow if h_inflow is not None else h0

        Q = h0.function_space()
        h, φ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)

        n = self.facet_normal(Q.mesh())
        outflow = firedrake.max_value(inner(u, n), 0)
        inflow = firedrake.min_value(inner(u, n), 0)

        flux_cells = -h * inner(u, grad(φ)) * dx
        flux_cells_lax = 0.5 * dt * div(h * u) * inner(u, grad(φ)) * dx
        flux_out = (h - 0.5 * dt * div(h * u)) * φ * outflow * ds
        F = h * φ * dx + dt * (flux_cells + flux_cells_lax + flux_out)

        accumulation = a * φ * dx
        flux_in = -(h_inflow - 0.5 * dt * div(h0 * u)) * φ * inflow * ds
        A = h0 * φ * dx + dt * (accumulation + flux_in)

        h = h0.copy(deepcopy=True)
        solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
        firedrake.solve(F == A, h, solver_parameters=solver_parameters)

        return h
