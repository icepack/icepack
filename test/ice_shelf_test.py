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
from firedrake import interpolate, as_vector
import icepack, icepack.models
from icepack.constants import gravity, rho_ice, rho_water, glen_flow_law as n

# The domain is a 20km x 20km square, with ice flowing in from the left.
L, W = 20.0e3, 20.0e3

# The inflow velocity is 100 m/year; the ice shelf decreases from 500m thick
# to 100m; and the temperature is a constant -19C.
u0 = 100.0
h0, dh = 500.0, 100.0
T = 254.15

# This is an exact solution for the velocity of a floating ice shelf with
# constant temperature and linearly decreasing thickness. See Greve and
# Blatter for the derivation.
def exact_u(x):
    rho = rho_ice * (1 - rho_ice / rho_water)
    Z = icepack.rate_factor(T) * (rho * gravity * h0 / 4)**n
    q = 1 - (1 - (dh/h0) * (x/L))**(n + 1)
    du = Z * q * L * (h0/dh) / (n + 1)
    return u0 + du

# We'll use the same perturbation to `u` throughout these tests.
def perturb_u(x, y):
    px, py = x/L, y/W
    q = 16 * px * (1 - px) * py * (1 - py)
    return 60 * q * (px - 0.5)

# Check that the diagnostic solver converges with the expected rate as the
# mesh is refined using an exact solution of the ice shelf model.
def test_diagnostic_solver_convergence():
    # Create an ice shelf model
    ice_shelf = icepack.models.IceShelf()
    opts = {'dirichlet_ids': [1, 3, 4], 'tol': 1e-12}

    # Solve the ice shelf model for successively higher mesh resolution
    delta_x, error = [], []
    norm = lambda v: icepack.norm(v, norm_type='H1')
    for N in range(16, 97, 4):
        mesh = firedrake.RectangleMesh(N, N, L, W)
        x, y = firedrake.SpatialCoordinate(mesh)

        degree = 2
        V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
        Q = firedrake.FunctionSpace(mesh, 'CG', degree)

        u_exact = interpolate(as_vector((exact_u(x), 0)), V)
        u_guess = interpolate(as_vector((exact_u(x) + perturb_u(x, y), 0)), V)

        h = interpolate(h0 - dh * x / L, Q)
        A = interpolate(firedrake.Constant(icepack.rate_factor(T)), Q)

        u = ice_shelf.diagnostic_solve(h=h, A=A, u0=u_guess, **opts)
        error.append(norm(u_exact - u) / norm(u_exact))
        delta_x.append(L / N)

        print(delta_x[-1], error[-1])

    # Fit the error curve and check that the convergence rate is what we
    # expect
    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print(slope, intercept)
    assert slope > degree - 0.05


# Check that the diagnostic solver converges with the expected rate as the
# mesh is refined when we use an alternative parameterization of the model.
def test_diagnostic_solver_alternate_parameterization():
    # Define a new viscosity functional, parameterized in terms of the
    # rheology `B` instead of the fluidity `A`
    from firedrake import inner, grad, sym, dx, tr as trace, Identity, sqrt
    def M(eps, B):
        I = Identity(2)
        tr = trace(eps)
        eps_e = sqrt((inner(eps, eps) + tr**2) / 2)
        mu = 0.5 * B * eps_e**(1/n - 1)
        return 2 * mu * (eps + tr * I)

    def eps(u):
        return sym(grad(u))

    def viscosity(u=None, h=None, B=None):
        return n/(n + 1) * h * inner(M(eps(u), B), eps(u)) * dx

    # Make a model object with our new viscosity functional
    ice_shelf = icepack.models.IceShelf(viscosity=viscosity)
    opts = {'dirichlet_ids': [1, 3, 4], 'tol': 1e-12}

    # Same as before
    delta_x, error = [], []
    norm = lambda v: icepack.norm(v, norm_type='H1')
    for N in range(16, 65, 4):
        mesh = firedrake.RectangleMesh(N, N, L, W)
        x, y = firedrake.SpatialCoordinate(mesh)

        degree = 2
        V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
        Q = firedrake.FunctionSpace(mesh, 'CG', degree)

        u_exact = interpolate(as_vector((exact_u(x), 0)), V)
        u_guess = interpolate(as_vector((exact_u(x) + perturb_u(x, y), 0)), V)
        h = interpolate(h0 - dh * x / L, Q)
        B = interpolate(firedrake.Constant(icepack.rate_factor(T)**(-1/n)), Q)

        u = ice_shelf.diagnostic_solve(h=h, B=B, u0=u_guess, **opts)
        error.append(norm(u_exact - u) / norm(u_exact))
        delta_x.append(L / N)

        print(delta_x[-1], error[-1])

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print(slope, intercept)
    assert slope > degree - 0.05

