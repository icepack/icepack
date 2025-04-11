# Copyright (C) 2017-2025 by Daniel Shapero <shapero@uw.edu>
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


# Test that our supposed analytical solution really is a solution using the
# symbolic algebra package `sympy`.
def test_manufactured_solution():
    from sympy import symbols, simplify, diff, lambdify
    from icepack import constants

    L, ρ_I, ρ_W = symbols("L rho_I rho_W", real=True, positive=True)
    A, g = symbols("A g", real=True, positive=True)
    n, m = symbols("n m", integer=True, positive=True)

    def membrane_stress(x, u, A):
        return simplify(2 * A ** (-1 / n) * diff(u, x) ** (1 / n))

    def friction(x, u, C):
        return simplify(-C * u ** (1 / m))

    def driving_stress(x, h, s):
        return simplify(-ρ_I * g * h * diff(s, x))

    def shelfy_stream_eqns(x, u, h, s, A, C):
        return simplify(
            diff(h * membrane_stress(x, u, A), x)
            + friction(x, u, C)
            + driving_stress(x, h, s)
        )

    def boundary_condition(x, u, h, s, A):
        M = membrane_stress(x, u, A)
        d = (s - h).subs(x, L)
        τ = (ρ_I * g * h**2 - ρ_W * g * d**2) / 2
        return simplify((h * M - τ).subs(x, L))

    x = symbols("x", real=True)

    h0, dh = symbols("h0 dh", real=True, positive=True)
    h = h0 - dh * x / L

    s0, ds = symbols("s0 ds", real=True, positive=True)
    s = s0 - ds * x / L

    h_L = h0 - dh
    s_L = s0 - ds
    β = dh / ds * (ρ_I * h_L ** 2 - ρ_W * (s_L - h_L) ** 2) / (ρ_I * h_L**2)

    ρ = β * ρ_I * ds / dh
    P = ρ * g * h / 4
    dP = ρ * g * dh / 4
    P0 = ρ * g * h0 / 4

    u0 = symbols("u0", real=True, positive=True)
    du = L * A * (P0 ** (n + 1) - P ** (n + 1)) / ((n + 1) * dP)
    u = u0 + du

    C = (1 - β) * (ρ_I * g * h) * ds / L * u ** (-1 / m)

    T = 254.15
    values = {
        u0: 100,
        h0: 500,
        dh: 100,
        s0: 150,
        ds: 90,
        L: 20e3,
        A: icepack.rate_factor(T),
        ρ_I: constants.ice_density,
        ρ_W: constants.water_density,
        n: constants.glen_flow_law,
        m: constants.weertman_sliding_law,
        g: constants.gravity,
    }

    τ_b = lambdify(x, friction(x, u, C).subs(values), "numpy")
    τ_d = lambdify(x, driving_stress(x, h, s).subs(values), "numpy")
    M = membrane_stress(x, u, A)
    τ_m = lambdify(x, simplify(diff(h * M, x)).subs(values), "numpy")
    xs = np.linspace(0, values[L], 21)

    tolerance = 1e-8
    assert abs(boundary_condition(x, u, h, s, A).subs(values)) < tolerance
    assert np.max(np.abs(τ_m(xs) + τ_b(xs) + τ_d(xs))) < tolerance * np.max(
        np.abs(τ_d(xs))
    )


# Now test our numerical solvers against this analytical solution.
import firedrake
from firedrake import Function, as_vector
import icepack
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    glen_flow_law as n,
    weertman_sliding_law as m,
    gravity as g,
)

Lx, Ly = 20e3, 20e3
h0, dh = 500.0, 100.0
s0, ds = 150.0, 90.0
T = 254.15
u_inflow = 100.0

# Pick the fraction of the gravitational driving that membrane stress
# divergence will take up.
h_L = h0 - dh
s_L = s0 - ds
β = dh / ds * (ρ_I * h_L**2 - ρ_W * (s_L - h_L)**2) / (ρ_I * h_L**2)

# We'll arbitrarily pick this to be the velocity, then we'll find a friction
# coefficient that makes this velocity an exact solution of the momentum
# balance equations.
def exact_u(x):
    A = icepack.rate_factor(T)
    ρ = β * ρ_I * ds / dh
    h = h0 - dh * x / Lx
    P = ρ * g * h / 4
    dP = ρ * g * dh / 4
    P0 = ρ * g * h0 / 4
    du = Lx * A * (P0 ** (n + 1) - P ** (n + 1)) / ((n + 1) * dP)
    return u_inflow + du


def perturb_u(x, y):
    px, py = x / Lx, y / Ly
    q = 16 * px * (1 - px) * py * (1 - py)
    return 60 * q * (px - 0.5)


def friction(x):
    h = h0 - dh * x / Lx
    return (1 - β) * (ρ_I * g * h) * ds / Lx * exact_u(x) ** (-1 / m)


def norm(v):
    return icepack.norm(v, norm_type="H1")


def test_diagnostic_solver_convergence():
    model = icepack.models.IceStream()
    opts = {"dirichlet_ids": [1], "side_wall_ids": [3, 4]}

    for degree in range(1, 4):
        delta_x, error = [], []
        for N in range(16, 97 - 32 * (degree - 1), 4):
            mesh = firedrake.RectangleMesh(N, N, Lx, Ly)
            x, y = firedrake.SpatialCoordinate(mesh)

            Q = firedrake.FunctionSpace(mesh, "CG", degree)
            V = firedrake.VectorFunctionSpace(mesh, "CG", degree)

            u_exact = Function(V).interpolate(as_vector((exact_u(x), 0)))
            u_guess = Function(V).interpolate(u_exact + as_vector((perturb_u(x, y), 0)))

            h = Function(Q).interpolate(h0 - dh * x / Lx)
            s = Function(Q).interpolate(s0 - ds * x / Lx)
            C = Function(Q).interpolate(friction(x))
            A = Function(Q).assign(firedrake.Constant(icepack.rate_factor(T)))

            solver = icepack.solvers.FlowSolver(model, **opts)
            u = solver.diagnostic_solve(
                velocity=u_guess,
                thickness=h,
                surface=s,
                fluidity=A,
                friction=C,
                strain_rate_min=firedrake.Constant(0.0),
            )
            error.append(norm(u_exact - u) / norm(u_exact))
            delta_x.append(Lx / N)

        log_delta_x = np.log2(np.array(delta_x))
        log_error = np.log2(np.array(error))
        slope, intercept = np.polyfit(log_delta_x, log_error, 1)

        print(f"log(error) ~= {slope:g} * log(dx) {intercept:+g}")
        assert slope > degree + 0.9


def test_computing_surface():
    N = 16
    mesh = firedrake.RectangleMesh(N, N, Lx, Ly)
    degree = 2
    Q = firedrake.FunctionSpace(mesh, "CG", degree)

    x, y = firedrake.SpatialCoordinate(mesh)
    h = Function(Q).interpolate(h0 - dh * x / Lx)
    b0 = ρ_I / ρ_W * (dh / 2 - h0)
    b = Function(Q).assign(firedrake.Constant(b0))

    s = icepack.compute_surface(thickness=h, bed=b)
    x0, y0 = Lx / 2, Ly / 2
    assert abs(s((x0, y0)) - (1 - ρ_I / ρ_W) * h((x0, y0))) < 1e-8
