# Copyright (C) 2019-2020 by Daniel Shapero <shapero@uw.edu>
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
import icepack
from icepack.constants import (ice_density as ρ_I, water_density as ρ_W,
                               gravity as g, glen_flow_law as n)

Lx = 20e3
h0, dh = 500.0, 100.0
T = 254.15
u_inflow = 100.0

height_above_flotation = 10.0
d = -ρ_I / ρ_W * (h0 - dh) + height_above_flotation
ρ = ρ_I - ρ_W * d**2 / (h0 - dh)**2

β = 1/2
α = β * ρ / ρ_I * dh / Lx
ds = (1 + β) * ρ / ρ_I * dh

def exact_u(x):
    Z = icepack.rate_factor(T) * (ρ * g * h0 / 4)**n
    q = 1 - (1 - (dh/h0) * (x/Lx))**(n + 1)
    du = Z * q * Lx * (h0/dh) / (n + 1)
    return u_inflow + du


def test_order_0():
    def h_expr(x):
        return h0 - dh * x / Lx

    def s_expr(x):
        return d + h0 - dh + ds * (1 - x / Lx)

    A = firedrake.Constant(icepack.rate_factor(254.15))
    C = firedrake.Constant(0.001)
    opts = {'dirichlet_ids': [1], 'tolerance': 1e-14}

    Nx = 64
    mesh1d = firedrake.IntervalMesh(Nx, Lx)
    x = firedrake.SpatialCoordinate(mesh1d)[0]
    Q1d = firedrake.FunctionSpace(mesh1d, family='CG', degree=2)
    h = firedrake.interpolate(h_expr(x), Q1d)
    s = firedrake.interpolate(s_expr(x), Q1d)
    u0 = firedrake.interpolate(exact_u(x), Q1d)

    model1d = icepack.models.IceStream(dimension=1)
    solver1d = icepack.solvers.FlowSolver(model1d, **opts)
    u1d = solver1d.diagnostic_solve(
         velocity=u0,
         thickness=h,
         surface=s,
         fluidity=A,
         friction=C
        )

    mesh = firedrake.ExtrudedMesh(mesh1d, layers=1)
    x, ζ = firedrake.SpatialCoordinate(mesh)
    Q2d = firedrake.FunctionSpace(
        mesh, family='CG', degree=2, vfamily='DG', vdegree=0)
    V2d = firedrake.FunctionSpace(
        mesh, family='CG', degree=2, vfamily='DG', vdegree=2)

    h = firedrake.interpolate(h_expr(x), Q2d)
    s = firedrake.interpolate(s_expr(x), Q2d)
    u0 = firedrake.interpolate(exact_u(x), V2d)

    model2d = icepack.models.HybridModel(dimension=1.5)
    solver2d = icepack.solvers.FlowSolver(model2d, **opts)
    u2d = solver2d.diagnostic_solve(
        velocity=u0,
        thickness=h,
        surface=s,
        fluidity=A,
        friction=C
    )

    U1D, U2D = u1d.dat.data_ro, u2d.dat.data_ro
    assert np.linalg.norm(U2D - U1D) / np.linalg.norm(U1D) < 1e-2


def test_diagnostic_solver():
    Nx = 32
    mesh1d = firedrake.IntervalMesh(Nx, Lx)
    mesh = firedrake.ExtrudedMesh(mesh1d, layers=1)
    Q = firedrake.FunctionSpace(
        mesh, family='CG', degree=2, vfamily='DG', vdegree=0)

    x, ζ = firedrake.SpatialCoordinate(mesh)
    h = firedrake.interpolate(h0 - dh * x / Lx, Q)
    s = firedrake.interpolate(d + h0 - dh + ds * (1 - x / Lx), Q)
    u_expr = (0.95 + 0.05 * ζ) * exact_u(x)

    A = firedrake.Constant(icepack.rate_factor(254.15))
    C = firedrake.Constant(0.001)

    model = icepack.models.HybridModel(dimension=1.5)
    opts = {'dirichlet_ids': [1], 'tol': 1e-12}

    max_degree = 5
    Nz = 32
    xs = np.array([(Lx/2, k / Nz) for k in range(Nz + 1)])
    us = np.zeros((max_degree + 1, Nz + 1))
    for vdegree in range(max_degree, 0, -1):
        solver = icepack.solvers.FlowSolver(model, **opts)
        V = firedrake.FunctionSpace(
            mesh, family='CG', degree=2, vfamily='DG', vdegree=vdegree)

        u0 = firedrake.interpolate(u_expr, V)
        u = solver.diagnostic_solve(
            velocity=u0,
            thickness=h,
            surface=s,
            fluidity=A,
            friction=C
        )

        V0 = firedrake.FunctionSpace(
            mesh, family='CG', degree=2, vfamily='DG', vdegree=0)

        depth_avg_u = firedrake.project(u, V0)
        shear_u = firedrake.project(u - depth_avg_u, V)
        assert icepack.norm(shear_u, norm_type='Linfty') > 1e-2

        us_center = np.array(u.at(xs, tolerance=1e-6))
        us[vdegree,:] = us_center

        norm = np.linalg.norm(us[max_degree, :])
        error = np.linalg.norm(us[vdegree, :] - us[max_degree, :]) / norm
        print(error, flush=True)
        assert error < 1e-2
