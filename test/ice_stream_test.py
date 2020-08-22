# Copyright (C) 2017-2020 by Daniel Shapero <shapero@uw.edu>
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
    L, ρ_I, ρ_W = symbols('L rho_I rho_W', real=True, positive=True)
    B, g = symbols('B g', real=True, positive=True)
    n, m = symbols('n m', integer=True, positive=True)

    def membrane_stress(x, u, B):
        return simplify(2 * B * diff(u, x)**(1/n))

    def friction(x, u, C):
        return simplify(-C * u**(1/m))

    def driving_stress(x, h, s):
        return simplify(-ρ_I * g * h * diff(s, x))

    def shelfy_stream_eqns(x, u, h, s, B, C):
        return simplify(diff(h * membrane_stress(x, u, B), x) +
                        friction(x, u, C) +
                        driving_stress(x, h, s))

    def boundary_condition(x, u, h, s, B):
        M = membrane_stress(x, u, B)
        d = (s - h).subs(x, L)
        τ = (ρ_I * g * h**2 - ρ_W * g * d**2) / 2
        return simplify((h * M - τ).subs(x, L))

    x = symbols('x', real=True)

    h0, dh = symbols('h0 dh', real=True, positive=True)
    h = h0 - dh * x / L

    hf = symbols('hf', real=True, positive=True)
    d = -ρ_I / ρ_W * h.subs(x, L) + hf
    ρ = (ρ_I - ρ_W * d**2 / h**2).subs(x, L)

    u0 = symbols('u0', real=True, positive=True)
    du = (ρ*g*h0/(4*B))**n * (1 - (1 - dh/h0*x/L)**(n+1)) * L * (h0/dh)/(n+1)
    u = u0 + du

    β = 1/2
    α = β * ρ / ρ_I * dh / L
    C = α * (ρ_I * g * h) * u**(-1/m)

    ds = (1 + β) * ρ / ρ_I * dh
    s = d + h.subs(x, L) + ds * (1 - x / L)

    T = 254.15
    rheology = icepack.rate_factor(T)**(-1/constants.glen_flow_law)
    values = {
        u0: 100, dh: 100, h0: 500, L: 20e3, hf: 10, B: rheology,
        ρ_I: constants.ice_density, ρ_W: constants.water_density,
        n: constants.glen_flow_law, m: constants.weertman_sliding_law,
        g: constants.gravity
    }

    τ_b = lambdify(x, friction(x, u, C).subs(values), 'numpy')
    τ_d = lambdify(x, driving_stress(x, h, s).subs(values), 'numpy')
    M = membrane_stress(x, u, B)
    τ_m = lambdify(x, simplify(diff(h * M, x)).subs(values), 'numpy')
    xs = np.linspace(0, values[L], 21)

    tolerance = 1e-8
    assert abs(boundary_condition(x, u, h, s, B).subs(values)) < tolerance
    assert (np.max(np.abs(τ_m(xs) + τ_b(xs) + τ_d(xs)))
            < tolerance * np.max(np.abs(τ_d(xs))))


# Now test our numerical solvers against this analytical solution.
import firedrake
from firedrake import interpolate, as_vector
import icepack
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    glen_flow_law as n,
    weertman_sliding_law as m,
    gravity as g
)

Lx, Ly = 20e3, 20e3
h0, dh = 500.0, 100.0
T = 254.15
u_inflow = 100.0

# Pick a height above flotation at the ice terminus. In order to have an
# exact ice velocity of the same form as the exact solution for an ice
# shelf, we have to pick the pseudo-density to be a certain value for the
# velocity to satisfy the boundary condition at the terminus.
height_above_flotation = 10.0
d = -ρ_I / ρ_W * (h0 - dh) + height_above_flotation
ρ = ρ_I - ρ_W * d**2 / (h0 - dh)**2


# We'll arbitrarily pick this to be the velocity, then we'll find a
# friction coefficient and surface elevation that makes this velocity
# an exact solution of the shelfy stream equations.
def exact_u(x):
    Z = icepack.rate_factor(T) * (ρ * g * h0 / 4)**n
    q = 1 - (1 - (dh/h0) * (x/Lx))**(n + 1)
    du = Z * q * Lx * (h0/dh) / (n + 1)
    return u_inflow + du


def perturb_u(x, y):
    px, py = x/Lx, y/Ly
    q = 16 * px * (1 - px) * py * (1 - py)
    return 60 * q * (px - 0.5)


# With this choice of friction coefficient, we can take the surface
# elevation to be a linear function of the horizontal coordinate and the
# velocity will be an exact solution of the shelfy stream equations.
β = 1/2
α = β * ρ / ρ_I * dh / Lx


def friction(x):
    return α * (ρ_I * g * (h0 - dh * x/Lx)) * exact_u(x)**(-1/m)


# Total change of the surface elevation
ds = (1 + β) * ρ / ρ_I * dh


def norm(v):
    return icepack.norm(v, norm_type='H1')


def test_diagnostic_solver_convergence():
    model = icepack.models.IceStream()
    opts = {'dirichlet_ids': [1], 'side_wall_ids': [3, 4]}

    for degree in range(1, 4):
        delta_x, error = [], []
        for N in range(16, 97 - 32 * (degree - 1), 4):
            mesh = firedrake.RectangleMesh(N, N, Lx, Ly)
            x, y = firedrake.SpatialCoordinate(mesh)

            Q = firedrake.FunctionSpace(mesh, 'CG', degree)
            V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)

            u_exact = interpolate(as_vector((exact_u(x), 0)), V)
            u_guess = interpolate(u_exact + as_vector((perturb_u(x, y), 0)), V)

            h = interpolate(h0 - dh * x/Lx, Q)
            s = interpolate(d + h0 - dh + ds * (1 - x / Lx), Q)
            C = interpolate(friction(x), Q)
            A = interpolate(firedrake.Constant(icepack.rate_factor(T)), Q)

            solver = icepack.solvers.FlowSolver(model, **opts)
            u = solver.diagnostic_solve(
                velocity=u_guess,
                thickness=h,
                surface=s,
                fluidity=A,
                friction=C
            )
            error.append(norm(u_exact - u) / norm(u_exact))
            delta_x.append(Lx / N)

            print(delta_x[-1], error[-1])

        log_delta_x = np.log2(np.array(delta_x))
        log_error = np.log2(np.array(error))
        slope, intercept = np.polyfit(log_delta_x, log_error, 1)

        print('log(error) ~= {:g} * log(dx) + {:g}'.format(slope, intercept))
        assert slope > degree + 0.9


def test_computing_surface():
    N = 16
    mesh = firedrake.RectangleMesh(N, N, Lx, Ly)
    degree = 2
    Q = firedrake.FunctionSpace(mesh, 'CG', degree)

    x, y = firedrake.SpatialCoordinate(mesh)
    h = interpolate(h0 - dh * x / Lx, Q)
    b0 = ρ_I / ρ_W * (dh / 2 - h0)
    b = interpolate(firedrake.Constant(b0), Q)

    s = icepack.compute_surface(h=h, b=b)
    x0, y0 = Lx/2, Ly/2
    assert abs(s((x0, y0)) - (1 - ρ_I / ρ_W) * h((x0, y0))) < 1e-8
