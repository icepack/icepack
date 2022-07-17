# Copyright (C) 2019-2021 by Daniel Shapero <shapero@uw.edu>
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
from firedrake import Constant
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
    h = firedrake.interpolate(h_expr(x), Q_x)
    s = firedrake.interpolate(s_expr(x), Q_x)
    if dim == "xz":
        u0 = firedrake.interpolate(exact_u(x), Q_x)
    elif dim == "xyz":
        V_x = firedrake.VectorFunctionSpace(mesh_x, "CG", 2)
        u_expr = firedrake.as_vector((exact_u(x), 0))
        u0 = firedrake.interpolate(u_expr, V_x)

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
        u0 = firedrake.interpolate(exact_u(x), V_xz)
    elif dim == "xyz":
        x, y, ζ = firedrake.SpatialCoordinate(mesh)
        V_xz = firedrake.VectorFunctionSpace(
            mesh, "CG", 2, vfamily="GL", vdegree=0, dim=2
        )
        u_expr = firedrake.as_vector((exact_u(x), 0))
        u0 = firedrake.interpolate(u_expr, V_xz)
    Q_xz = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="DG", vdegree=0)
    h = firedrake.interpolate(h_expr(x), Q_xz)
    s = firedrake.interpolate(s_expr(x), Q_xz)

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

    h = firedrake.interpolate(h0 - dh * x / Lx, Q)
    s = firedrake.interpolate(d + h0 - dh + ds * (1 - x / Lx), Q)

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

        u0 = firedrake.interpolate(u_expr, V)
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
