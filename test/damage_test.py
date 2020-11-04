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

import firedrake
from firedrake import norm, interpolate, Constant, as_vector, sym, grad
import icepack
from icepack.models.viscosity import membrane_stress
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g, glen_flow_law as n
)
from icepack.utilities import eigenvalues

def test_eigenvalues():
    nx, ny = 32, 32
    mesh = firedrake.UnitSquareMesh(nx, ny)
    x, y = firedrake.SpatialCoordinate(mesh)

    V = firedrake.VectorFunctionSpace(mesh, family='CG', degree=2)
    u = interpolate(as_vector((x, 0)), V)

    Q = firedrake.FunctionSpace(mesh, family='DG', degree=2)
    ε = sym(grad(u))
    Λ1, Λ2 = eigenvalues(ε)
    λ1 = firedrake.project(Λ1, Q)
    λ2 = firedrake.project(Λ2, Q)

    assert norm(λ1 - Constant(1)) < norm(u) / (nx * ny)
    assert norm(λ2) < norm(u) / (nx * ny)


def test_damage_transport():
    nx, ny = 32, 32
    Lx, Ly = 20e3, 20e3
    mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly)
    x, y = firedrake.SpatialCoordinate(mesh)

    V = firedrake.VectorFunctionSpace(mesh, 'CG', 2)
    Q = firedrake.FunctionSpace(mesh, 'CG', 2)

    u0 = 100.0
    h0, dh = 500.0, 100.0
    T = 268.0

    ρ = ρ_I * (1 - ρ_I / ρ_W)
    Z = icepack.rate_factor(T) * (ρ * g * h0 / 4)**n
    q = 1 - (1 - (dh / h0) * (x / Lx))**(n + 1)
    du = Z * q * Lx * (h0 / dh) / (n + 1)

    u = interpolate(as_vector((u0 + du, 0)), V)
    h = interpolate(h0 - dh * x / Lx, Q)
    A = firedrake.Constant(icepack.rate_factor(T))

    S = firedrake.TensorFunctionSpace(mesh, 'DG', 1)
    ε = firedrake.project(sym(grad(u)), S)
    M = firedrake.project(membrane_stress(ε, A), S)

    degree = 1
    Δ = firedrake.FunctionSpace(mesh, 'DG', degree)
    D_inflow = firedrake.Constant(0.0)
    D = firedrake.Function(Δ)

    damage_model = icepack.models.DamageTransport()
    damage_solver = icepack.solvers.DamageSolver(damage_model)

    final_time = Lx / u0
    max_speed = u.at((Lx - 1., Ly / 2), tolerance=1e-10)[0]
    δx = Lx / nx
    timestep = δx / max_speed / (2 * degree + 1)
    num_steps = int(final_time / timestep)
    dt = final_time / num_steps

    for step in range(num_steps):
        D = damage_solver.solve(
            dt,
            damage=D,
            velocity=u,
            strain_rate=ε,
            membrane_stress=M,
            damage_inflow=D_inflow
        )

    Dmax = D.dat.data_ro[:].max()
    assert 0 < Dmax < 1
