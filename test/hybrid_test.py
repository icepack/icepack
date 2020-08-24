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
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    glen_flow_law as n,
    weertman_sliding_law as m
)

Lx, Ly = 20e3, 20e3
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


# Check that using degree-0 elements in the vertical gives the same answer as
# the ice stream model
def test_order_0():
    def h_expr(x):
        return h0 - dh * x / Lx

    def s_expr(x):
        return d + h0 - dh + ds * (1 - x / Lx)

    A = firedrake.Constant(icepack.rate_factor(254.15))
    C = firedrake.Constant(0.001)
    opts = {'dirichlet_ids': [1], 'side_wall_ids': [3, 4], 'tolerance': 1e-14}

    Nx, Ny = 64, 64
    mesh2d = firedrake.RectangleMesh(Nx, Ny, Lx, Ly)
    x, y = firedrake.SpatialCoordinate(mesh2d)
    Q2d = firedrake.FunctionSpace(mesh2d, family='CG', degree=2)
    V2d = firedrake.VectorFunctionSpace(mesh2d, family='CG', degree=2)
    h = firedrake.interpolate(h_expr(x), Q2d)
    s = firedrake.interpolate(s_expr(x), Q2d)
    u_expr = firedrake.as_vector((exact_u(x), 0))
    u0 = firedrake.interpolate(u_expr, V2d)

    model2d = icepack.models.IceStream()
    solver2d = icepack.solvers.FlowSolver(model2d, **opts)
    u2d = solver2d.diagnostic_solve(
        velocity=u0,
        thickness=h,
        surface=s,
        fluidity=A,
        friction=C
    )

    mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)
    x, y, ζ = firedrake.SpatialCoordinate(mesh)
    Q3d = firedrake.FunctionSpace(
        mesh, family='CG', degree=2, vfamily='DG', vdegree=0)
    V3d = firedrake.VectorFunctionSpace(
        mesh, dim=2, family='CG', degree=2, vfamily='GL', vdegree=0)
    h = firedrake.interpolate(h_expr(x), Q3d)
    s = firedrake.interpolate(s_expr(x), Q3d)
    u_expr = firedrake.as_vector((exact_u(x), 0))
    u0 = firedrake.interpolate(u_expr, V3d)

    model3d = icepack.models.HybridModel()
    solver3d = icepack.solvers.FlowSolver(model3d, **opts)
    u3d = solver3d.diagnostic_solve(
        velocity=u0,
        thickness=h,
        surface=s,
        fluidity=A,
        friction=C
    )

    U2D, U3D = u2d.dat.data_ro, u3d.dat.data_ro
    assert np.linalg.norm(U3D - U2D) / np.linalg.norm(U2D) < 1e-2


# Check that the hybrid model has non-trivial vertical shear and that the depth
# averages roughly agree for different vertical degrees
def test_diagnostic_solver():
    Nx, Ny = 32, 32
    mesh2d = firedrake.RectangleMesh(Nx, Ny, Lx, Ly)
    mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)
    Q = firedrake.FunctionSpace(
        mesh, family='CG', degree=2, vfamily='DG', vdegree=0)

    x, y, ζ = firedrake.SpatialCoordinate(mesh)
    h = firedrake.interpolate(h0 - dh * x / Lx, Q)
    s = firedrake.interpolate(d + h0 - dh + ds * (1 - x / Lx), Q)
    u_expr = firedrake.as_vector(((0.95 + 0.05 * ζ) * exact_u(x), 0))

    A = firedrake.Constant(icepack.rate_factor(254.15))
    C = firedrake.Constant(0.001)

    model = icepack.models.HybridModel()
    opts = {'dirichlet_ids': [1, 3, 4], 'tol': 1e-12}

    max_degree = 5
    Nz = 32
    xs = np.array([(Lx/2, Ly/2, k / Nz) for k in range(Nz + 1)])
    us = np.zeros((max_degree + 1, Nz + 1))
    for vdegree in range(max_degree, 0, -1):
        solver = icepack.solvers.FlowSolver(model, **opts)
        V = firedrake.VectorFunctionSpace(
            mesh, dim=2, family='CG', degree=2, vfamily='GL', vdegree=vdegree
        )
        u0 = firedrake.interpolate(u_expr, V)
        u = solver.diagnostic_solve(
            velocity=u0,
            thickness=h,
            surface=s,
            fluidity=A,
            friction=C
        )

        V0 = firedrake.VectorFunctionSpace(
            mesh, dim=2, family='CG', degree=2, vfamily='DG', vdegree=0
        )

        depth_avg_u = firedrake.project(u, V0)
        shear_u = firedrake.project(u - depth_avg_u, V)
        assert icepack.norm(shear_u, norm_type='Linfty') > 1e-2

        us_center = np.array(u.at(xs, tolerance=1e-6))
        us[vdegree,:] = np.sqrt(np.sum(us_center**2, 1))

        norm = np.linalg.norm(us[max_degree, :])
        error = np.linalg.norm(us[vdegree, :] - us[max_degree, :]) / norm
        print(error, flush=True)
        assert error < 1e-2


# Test solving the coupled diagnostic/prognostic equations for the hybrid flow
# model and check that it doesn't explode.
def test_hybrid_prognostic_solve():
    Lx, Ly = 20e3, 20e3
    h0, dh = 500.0, 100.0
    T = 254.15
    u_in = 100.0

    model = icepack.models.HybridModel()
    opts = {'dirichlet_ids': [1], 'side_wall_ids': [3, 4], 'tolerance': 1e-12}

    Nx, Ny = 32, 32
    mesh2d = firedrake.RectangleMesh(Nx, Ny, Lx, Ly)
    mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)

    V = firedrake.VectorFunctionSpace(mesh, dim=2, family='CG', degree=2,
                                      vfamily='GL', vdegree=1)
    Q = firedrake.FunctionSpace(mesh, family='CG', degree=2,
                                vfamily='DG', vdegree=0)

    x, y, ζ = firedrake.SpatialCoordinate(mesh)
    height_above_flotation = 10.0
    d = -ρ_I / ρ_W * (h0 - dh) + height_above_flotation
    ρ = ρ_I - ρ_W * d**2 / (h0 - dh)**2

    Z = icepack.rate_factor(T) * (ρ * g * h0 / 4)**n
    q = 1 - (1 - (dh/h0) * (x/Lx))**(n + 1)
    ux = u_in + Z * q * Lx * (h0/dh) / (n + 1)
    u0 = firedrake.interpolate(firedrake.as_vector((ux, 0)), V)

    thickness = h0 - dh * x / Lx
    β = 1/2
    α = β * ρ / ρ_I * dh / Lx
    h = firedrake.interpolate(h0 - dh * x / Lx, Q)
    h_inflow = h.copy(deepcopy=True)
    ds = (1 + β) * ρ / ρ_I * dh
    s = firedrake.interpolate(d + h0 - dh + ds * (1 - x / Lx), Q)
    b = firedrake.interpolate(s - h, Q)

    C = firedrake.interpolate(α * (ρ_I * g * thickness) * ux**(-1 / m), Q)
    A = firedrake.Constant(icepack.rate_factor(T))

    final_time = 1.0
    timestep = 2.0 / Nx
    num_timesteps = int(final_time / timestep)
    dt = final_time / num_timesteps

    solver = icepack.solvers.FlowSolver(model, **opts)
    u = solver.diagnostic_solve(
        velocity=u0, thickness=h, surface=s, fluidity=A, friction=C
    )

    # Make the accumulation so that the system is exactly in steady state
    a = firedrake.Function(Q)
    h_n = solver.prognostic_solve(
        dt, thickness=h, velocity=u, accumulation=a, thickness_inflow=h_inflow
    )
    a.interpolate((h_n - h) / dt)

    for k in range(num_timesteps):
        h = solver.prognostic_solve(
            dt,
            thickness=h,
            velocity=u,
            accumulation=a,
            thickness_inflow=h_inflow
        )
        s = icepack.compute_surface(thickness=h, bed=b)

        u = solver.diagnostic_solve(
            velocity=u,
            thickness=h,
            surface=s,
            fluidity=A,
            friction=C
        )

    assert icepack.norm(h, norm_type='Linfty') < np.inf
