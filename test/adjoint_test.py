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

import numpy as np
import firedrake
from firedrake import inner, grad, dx, ds, interpolate
import icepack, icepack.models, icepack.adjoint

def test_poisson_rhs():
    for N in range(32, 65, 4):
        mesh = firedrake.UnitSquareMesh(N, N)
        x, y = firedrake.SpatialCoordinate(mesh)

        degree = 2
        Q = firedrake.FunctionSpace(mesh, 'CG', degree)

        expr = x * (1 - x) * y * (1 - y)
        f = interpolate(expr, Q)

        u = firedrake.Function(Q)
        v = firedrake.TestFunction(Q)
        F = (inner(grad(u), grad(v)) - f * v) * dx

        bc = firedrake.DirichletBC(Q, 0, "on_boundary")
        firedrake.solve(F == 0, u, bc)

        J = 0.5 * u**2 * dx
        dJ = icepack.adjoint.derivative(J, F, u, f, bc)
        J0 = firedrake.assemble(J)

        num_samples = 10
        errors = np.zeros(num_samples)
        dJ_df = firedrake.assemble(firedrake.action(dJ, f))
        for n in range(num_samples):
            delta = 2**(-n)
            f.assign(interpolate((1 + delta) * expr, Q))
            firedrake.solve(F == 0, u, bc)
            errors[n] = abs(firedrake.assemble(J) - (J0 + delta * dJ_df))

        deltas = np.array([2**(-n) for n in range(num_samples)])
        slope, intercept = np.polyfit(np.log2(deltas), np.log2(errors), 1)
        print(slope, intercept)
        assert slope > 1.9


def test_poisson_conductivity():
    for N in range(32, 65, 4):
        mesh = firedrake.UnitSquareMesh(N, N)
        x, y = firedrake.SpatialCoordinate(mesh)

        degree = 2
        Q = firedrake.FunctionSpace(mesh, 'CG', degree)

        f = interpolate(x * (1 - x) * y * (1 - y), Q)
        a = interpolate(firedrake.Constant(1), Q)
        b = interpolate(firedrake.exp(-((x - 0.5)**2 + (y - 0.5)**2)/2) / 4, Q)

        u = firedrake.Function(Q)
        v = firedrake.TestFunction(Q)
        F = (a * inner(grad(u), grad(v)) - f * v) * dx

        bc = firedrake.DirichletBC(Q, 0, "on_boundary")
        firedrake.solve(F == 0, u, bc)

        J = 0.5 * u**2 * dx
        dJ = icepack.adjoint.derivative(J, F, u, a, bc)
        J0 = firedrake.assemble(J)

        num_samples = 12
        errors = np.zeros(num_samples)
        dJ_db = firedrake.assemble(firedrake.action(dJ, b))
        for n in range(num_samples):
            delta = 2**(-n)
            a.assign(1 + delta * b)
            firedrake.solve(F == 0, u, bc)
            errors[n] = abs(firedrake.assemble(J) - (J0 + delta * dJ_db))

        deltas = np.array([2**(-n) for n in range(num_samples)])
        slope, intercept = np.polyfit(np.log2(deltas), np.log2(errors), 1)
        print(slope, intercept)
        assert slope > 1.9


def test_ice_shelf_rheology():
    L, W = 20e3, 20e3
    u_inflow = 100.0
    h0, dh = 500.0, 100.0
    T = 254.15

    from icepack.constants import rho_ice, rho_water, \
        glen_flow_law as n, gravity as g
    rho = rho_ice * (1 - rho_ice / rho_water)

    ice_shelf = icepack.models.IceShelf()
    opts = {'dirichlet_ids': [1, 3, 4], 'tol': 1e-12}

    for N in range(32, 65, 4):
        # Set up the mesh and function spaces
        mesh = firedrake.RectangleMesh(N, N, L, W)
        x, y = firedrake.SpatialCoordinate(mesh)

        degree = 2
        V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
        Q = firedrake.FunctionSpace(mesh, 'CG', degree)

        # Make the fields for the basic state
        Z = icepack.rate_factor(T) * (rho * g * h0 / 4)**n
        q = 1 - (1 - (dh/h0) * (x/L))**(n + 1)
        du = Z * q * L * (h0/dh) / (n + 1)
        u0 = interpolate(firedrake.as_vector((u_inflow + du, 0)), V)

        h = interpolate(h0 - dh * x / L, Q)
        A0 = interpolate(firedrake.Constant(icepack.rate_factor(T)), Q)
        u = ice_shelf.diagnostic_solve(u0=u0, h=h, A=A0, **opts)

        # Create the weak form of the PDE
        E = ice_shelf.action(u=u, h=h, A=A0, ice_front_ids=[2])
        F = firedrake.derivative(E, u)

        # Make the perturbation to the fluidity field
        delta_T = 5.0
        delta_A = icepack.rate_factor(T + delta_T) - icepack.rate_factor(T)
        px, py = x/L, y/W
        B = interpolate(16 * px * (1 - px) * py * (1 - py) * delta_A, Q)

        # Make the error functional and calculate its derivative
        nu = firedrake.FacetNormal(mesh)
        J = h * inner(u, nu) * ds(2)

        bcs = [firedrake.DirichletBC(V, (0, 0), k)
               for k in opts['dirichlet_ids']]
        dJ = icepack.adjoint.derivative(J, F, u, A0, bcs)
        J0 = firedrake.assemble(J)
        dJ_dB = firedrake.assemble(firedrake.action(dJ, B))

        num_samples = 8
        errors = np.zeros(num_samples)
        for k in range(num_samples):
            delta = 2**(-k)

            A = firedrake.Function(Q)
            A.assign(A0 + delta * B)
            u.assign(ice_shelf.diagnostic_solve(u0=u0, h=h, A=A, **opts))

            errors[k] = abs(firedrake.assemble(J) - (J0 + delta * dJ_dB))

        deltas = np.array([2**(-n) for n in range(num_samples)])
        slope, intercept = np.polyfit(np.log2(deltas), np.log2(errors), 1)
        print(slope, intercept)
        assert slope > 1.9

