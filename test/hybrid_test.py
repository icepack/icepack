# Copyright (C) 2019-2025 by Daniel Shapero <shapero@uw.edu>
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

import pytest
import firedrake
from firedrake import inner, grad, Constant
import icepack
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    glen_flow_law as n,
)
import numpy as np

Lx, Ly = 20e3, 20e3
h0, dh = 500.0, 100.0
T = 254.15
u_inflow = 100.0

height_above_flotation = 10.0
d = -ρ_I / ρ_W * (h0 - dh) + height_above_flotation
ρ = ρ_I - ρ_W * d**2 / (h0 - dh) ** 2

β = 1 / 2
α = β * ρ / ρ_I * dh / Lx
ds = (1 + β) * ρ / ρ_I * dh


def exact_u(x):
    Z = icepack.rate_factor(T) * (ρ * g * h0 / 4) ** n
    q = 1 - (1 - (dh / h0) * (x / Lx)) ** (n + 1)
    du = Z * q * Lx * (h0 / dh) / (n + 1)
    return u_inflow + du


@pytest.mark.parametrize("dim", ["xz", "xyz"])
def test_order_0(dim):
    def h_expr(x):
        return h0 - dh * x / Lx

    def s_expr(x):
        return d + h0 - dh + ds * (1 - x / Lx)

    A = Constant(icepack.rate_factor(254.15))
    C = Constant(0.001)

    Nx, Ny = 64, 64
    if dim == "xz":
        opts = {"dirichlet_ids": [1], "tolerance": 1e-14}
        mesh_x = firedrake.IntervalMesh(Nx, Lx)
        x = firedrake.SpatialCoordinate(mesh_x)[0]
    elif dim == "xyz":
        opts = {"dirichlet_ids": [1], "side_wall_ids": [3, 4], "tolerance": 1e-14}
        mesh_x = firedrake.RectangleMesh(Nx, Ny, Lx, Ly)
        x, y = firedrake.SpatialCoordinate(mesh_x)
    Q_x = firedrake.FunctionSpace(mesh_x, "CG", 2)
    h = firedrake.Function(Q_x).interpolate(h_expr(x))
    s = firedrake.Function(Q_x).interpolate(s_expr(x))
    if dim == "xz":
        u0 = firedrake.Function(Q_x).interpolate(exact_u(x))
    elif dim == "xyz":
        V_x = firedrake.VectorFunctionSpace(mesh_x, "CG", 2)
        u_expr = firedrake.as_vector((exact_u(x), 0))
        u0 = firedrake.Function(V_x).interpolate(u_expr)

    model_x = icepack.models.IceStream()
    solver_x = icepack.solvers.FlowSolver(model_x, **opts)
    u_x = solver_x.diagnostic_solve(
        velocity=u0,
        thickness=h,
        surface=s,
        fluidity=A,
        friction=C,
        strain_rate_min=Constant(0.0),
    )

    mesh = firedrake.ExtrudedMesh(mesh_x, layers=1)
    if dim == "xz":
        x, ζ = firedrake.SpatialCoordinate(mesh)
        V_xz = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=0)
        u0 = firedrake.Function(V_xz).interpolate(exact_u(x))
    elif dim == "xyz":
        x, y, ζ = firedrake.SpatialCoordinate(mesh)
        V_xz = firedrake.VectorFunctionSpace(
            mesh, "CG", 2, vfamily="GL", vdegree=0, dim=2
        )
        u_expr = firedrake.as_vector((exact_u(x), 0))
        u0 = firedrake.Function(V_xz).interpolate(u_expr)
    Q_xz = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="DG", vdegree=0)
    h = firedrake.Function(Q_xz).interpolate(h_expr(x))
    s = firedrake.Function(Q_xz).interpolate(s_expr(x))

    model_xz = icepack.models.HybridModel()
    solver_xz = icepack.solvers.FlowSolver(model_xz, **opts)
    u_xz = solver_xz.diagnostic_solve(
        velocity=u0,
        thickness=h,
        surface=s,
        fluidity=A,
        friction=C,
        strain_rate_min=Constant(0.0),
    )

    U_x, U_xz = u_x.dat.data_ro, u_xz.dat.data_ro
    assert np.linalg.norm(U_xz - U_x) / np.linalg.norm(U_x) < 1e-2


def test_sia_limit():
    Nx = 32
    mesh_x = firedrake.IntervalMesh(Nx, Lx)
    x, = firedrake.SpatialCoordinate(mesh_x)

    Q_x = firedrake.FunctionSpace(mesh_x, "CG", 2)
    h = firedrake.Function(Q_x).interpolate(h0 - dh * x / Lx)
    s0 = Constant(h0 / 2)
    δs = Constant((h0 - dh) / 2)
    s = firedrake.Function(Q_x).interpolate(s0 - δs * x / Lx)

    # Compute the SIA solution
    V_x = firedrake.FunctionSpace(mesh_x, "CG", 2)
    p = ρ_I * g * h
    A = Constant(icepack.rate_factor(273.0))
    u_expr = 2 * A / (n + 2) * p**n * h * (δs / Lx)**n
    u0 = firedrake.Function(V_x).interpolate(u_expr)

    def penalty(**kwargs):
        u = kwargs["velocity"]
        h = kwargs["thickness"]
        λ = Constant(0.0)
        return 0.5 * λ*2 * inner(grad(u), grad(u))

    sia_model = icepack.models.ShallowIce(penalty=penalty)
    sia_opts = {
        "dirichlet_ids": [1],
        "diagnostic_solver_parameters": {"snes_rtol": 1e-6},
    }
    sia_solver = icepack.solvers.FlowSolver(sia_model, **sia_opts)
    u_sia = sia_solver.diagnostic_solve(
        velocity=u0, thickness=h, surface=s, fluidity=A
    )

    mesh = firedrake.ExtrudedMesh(mesh_x, layers=1)
    x, ζ = firedrake.SpatialCoordinate(mesh)
    Q_xz = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
    h = firedrake.Function(Q_xz).interpolate(h0 - dh * x / Lx)
    s = firedrake.Function(Q_xz).interpolate(s0 - δs * x / Lx)

    V_xz = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=4)
    p = ρ_I * g * h
    u_expr = 2 * A / (n + 2) * p**n * h * (δs / Lx)**n
    u0 = firedrake.Function(V_xz).interpolate(u_expr)

    # Make the friction coefficient very large and compute the hybrid model
    # solution
    f = Constant(100.0)
    expr = f * (h * A) ** (-1 / n)
    C = firedrake.Function(Q_xz).interpolate(expr)
    model = icepack.models.HybridModel()
    opts = {"dirichlet_ids": [1, 2]}
    solver = icepack.solvers.FlowSolver(model, **opts)
    u = solver.diagnostic_solve(
        velocity=u0,
        thickness=h,
        surface=s,
        fluidity=A,
        friction=C,
        strain_rate_min=Constant(0.0),
    )
    u_avg = icepack.depth_average(u)

    # Check that the relative error isn't too large
    assert firedrake.norm(u_sia - u_avg) / firedrake.norm(u_sia) < 0.1


@pytest.mark.parametrize("dim", ["xz", "xyz"])
def test_diagnostic_solver(dim):
    Nx, Ny, Nz = 32, 32, 32
    if dim == "xz":
        opts = {"dirichlet_ids": [1], "tol": 1e-12}
        mesh_x = firedrake.IntervalMesh(Nx, Lx)
        mesh = firedrake.ExtrudedMesh(mesh_x, layers=1)
        x, ζ = firedrake.SpatialCoordinate(mesh)
        u_expr = (0.95 + 0.05 * ζ) * exact_u(x)
        xs = np.array([(Lx / 2, k / Nz) for k in range(Nz + 1)])
    elif dim == "xyz":
        opts = {"dirichlet_ids": [1, 3, 4], "tol": 1e-12}
        mesh_x = firedrake.RectangleMesh(Nx, Ny, Lx, Ly)
        mesh = firedrake.ExtrudedMesh(mesh_x, layers=1)
        x, y, ζ = firedrake.SpatialCoordinate(mesh)
        u_expr = firedrake.as_vector(((0.95 + 0.05 * ζ) * exact_u(x), 0))
        xs = np.array([(Lx / 2, Ly / 2, k / Nz) for k in range(Nz + 1)])

    Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="DG", vdegree=0)

    h = firedrake.Function(Q).interpolate(h0 - dh * x / Lx)
    s = firedrake.Function(Q).interpolate(d + h0 - dh + ds * (1 - x / Lx))

    A = Constant(icepack.rate_factor(254.15))
    C = Constant(0.001)

    model = icepack.models.HybridModel()

    max_degree = 5
    us = np.zeros((max_degree + 1, Nz + 1))
    for vdegree in range(max_degree, 0, -1):
        solver = icepack.solvers.FlowSolver(model, **opts)
        if dim == "xz":
            V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="DG", vdegree=vdegree)
            V0 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="DG", vdegree=0)
        elif dim == "xyz":
            V = firedrake.VectorFunctionSpace(
                mesh, "CG", 2, vfamily="DG", vdegree=vdegree, dim=2
            )
            V0 = firedrake.VectorFunctionSpace(
                mesh, "CG", 2, vfamily="DG", vdegree=0, dim=2
            )

        u0 = firedrake.Function(V).interpolate(u_expr)
        u = solver.diagnostic_solve(
            velocity=u0, thickness=h, surface=s, fluidity=A, friction=C
        )

        depth_avg_u = firedrake.project(u, V0)
        shear_u = firedrake.project(u - depth_avg_u, V)
        assert icepack.norm(shear_u, norm_type="Linfty") > 1e-2

        us_center = np.array(u.at(xs, tolerance=1e-6))
        if dim == "xz":
            us[vdegree, :] = us_center
        elif dim == "xyz":
            us[vdegree, :] = np.sqrt(np.sum(us_center**2, 1))

        norm = np.linalg.norm(us[max_degree, :])
        error = np.linalg.norm(us[vdegree, :] - us[max_degree, :]) / norm
        print(error, flush=True)
        assert error < 1e-2


def test_diagnostic_solver_side_friction():
    Nx, Nz = 32, 32
    opts = {"dirichlet_ids": [1], "tol": 1e-12}
    mesh_x = firedrake.IntervalMesh(Nx, Lx)
    mesh = firedrake.ExtrudedMesh(mesh_x, layers=1)
    x, ζ = firedrake.SpatialCoordinate(mesh)
    u_expr = (0.95 + 0.05 * ζ) * exact_u(x)

    Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="DG", vdegree=0)

    h = firedrake.Function(Q).interpolate(h0 - dh * x / Lx)
    s = firedrake.Function(Q).interpolate(d + h0 - dh + ds * (1 - x / Lx))

    A = Constant(icepack.rate_factor(254.15))
    C = Constant(0.001)

    model = icepack.models.HybridModel()

    vdegree = 2
    solver = icepack.solvers.FlowSolver(model, **opts)
    solver_cs = icepack.solvers.FlowSolver(model, **opts)
    V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="DG", vdegree=vdegree)
    V0 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="DG", vdegree=0)

    u0 = firedrake.Function(V).interpolate(u_expr)

    u = solver.diagnostic_solve(
        velocity=u0, thickness=h, surface=s, fluidity=A, friction=C
    )

    Cs = Constant(1.0e-7)
    u_cs = solver_cs.diagnostic_solve(
        velocity=u0, thickness=h, surface=s, fluidity=A, friction=C, side_friction=Cs
    )

    depth_avg_u = firedrake.project(u, V0)
    depth_avg_u_cs = firedrake.project(u_cs, V0)

    xs = np.array([(Lx / 2, k / Nz) for k in range(Nz + 1)])
    us_center = np.array(u.at(xs, tolerance=1e-6))
    us_cs_center = np.array(u_cs.at(xs, tolerance=1e-6))

    # Faster w/o sidewall drag
    assert icepack.norm(depth_avg_u, norm_type="Linfty") > icepack.norm(
        depth_avg_u_cs, norm_type="Linfty"
    )

    # More internal deformation w/o sidewall drag, not just more sliding
    assert (us_center[0] - us_center[-1]) > (us_cs_center[0] - us_cs_center[-1])
