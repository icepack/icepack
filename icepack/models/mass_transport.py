"""Solvers for the mass continuity equation

This module contains a solver for the conservative advection equation that
describes the evolution of ice thickness. While the basic mass transport
solver will suffice for, say, ice shelf flow, other models that describe
grounded glaciers will also need to update the ice surface elevation in a
manner consistent with the bed elevation and where the ice may go afloat.
"""

import firedrake as fe
from firedrake import grad, div, dx, ds, inner

class MassTransport(object):
    def solve(self, dt, h0=None, a=None, u=None, **kwargs):
        Q = h0.ufl_function_space()
        h, phi = fe.TrialFunction(Q), fe.TestFunction(Q)
        mesh = Q.mesh()

        def outflow(v, n):
            vn = inner(v, n)
            return fe.conditional(fe.gt(vn, 0), vn, 0)

        def inflow(v, n):
            vn = inner(v, n)
            return fe.conditional(fe.le(vn, 0), vn, 0)

        n = fe.FacetNormal(mesh)

        F = (h * (phi - dt * inner(u, grad(phi))) * dx
             + dt * h * phi * outflow(u, n) * ds)

        A = ((h0 + dt * a) * phi * dx
             - dt * h0 * phi * inflow(u, n) * ds)

        h = fe.Function(Q)
        h.assign(h0)
        fe.solve(F == A, h,
                 solver_parameters={'ksp_type': 'gmres', 'pc_type': 'ilu'})
        return h

