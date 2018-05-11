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

import math
import numpy as np
import firedrake
from firedrake import interpolate
import icepack
from icepack.models.mass_transport import MassTransport

# Test solving the mass transport equations with a constant velocity field
# and check that the solutions converge to the exact solution obtained from
# the method of characteristics.
def test_mass_transport_solver_convergence():
    Lx, Ly = 1.0, 1.0
    u0 = 1.0
    h_in, dh = 1.0, 0.2

    delta_x, error = [], []
    mass_transport = MassTransport()
    norm = lambda v: icepack.norm(v, norm_type='L1')
    for N in range(16, 97, 4):
        delta_x.append(Lx / N)

        mesh = firedrake.RectangleMesh(N, N, Lx, Ly)
        x, y = firedrake.SpatialCoordinate(mesh)

        degree = 1
        V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
        Q = firedrake.FunctionSpace(mesh, 'CG', degree)

        h = interpolate(h_in - dh * x / Lx, Q)
        a = firedrake.Function(Q)
        u = interpolate(firedrake.as_vector((u0, 0)), V)
        T = 0.5
        num_timesteps = math.ceil(0.5 * N * u0 * T / Lx)
        dt = T / num_timesteps

        for k in range(num_timesteps):
            h = mass_transport.solve(dt, h0=h, a=a, u=u)

        z = x - u0 * T
        h_exact = interpolate(h_in - dh/Lx * firedrake.max_value(0, z), Q)
        error.append(norm(h - h_exact) / norm(h_exact))

        print(delta_x[-1], error[-1])

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    assert slope > degree - 0.05
    print(slope, intercept)


from icepack.constants import rho_ice, rho_water, \
    gravity as g, glen_flow_law as n

# Test solving the coupled diagnostic/prognostic equations for an ice shelf
# with thickness and velocity fields that are exactly insteady state.
def test_ice_shelf_prognostic_solver():
    from icepack.models import IceShelf
    rho = rho_ice * (1 - rho_ice/rho_water)

    Lx, Ly = 20.0e3, 20.0e3
    h0, dh = 500.0, 100.0
    u0 = 100.0
    T = 254.15

    ice_shelf = IceShelf()
    opts = {'dirichlet_ids': [1, 3, 4], 'tol': 1e-12}
    norm = lambda h: icepack.norm(h, norm_type='L1')

    delta_x, error = [], []
    for N in range(16, 65, 4):
        delta_x.append(Lx / N)

        mesh = firedrake.RectangleMesh(N, N, Lx, Ly)
        x, y = firedrake.SpatialCoordinate(mesh)

        degree = 2
        V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
        Q = firedrake.FunctionSpace(mesh, 'CG', degree)

        q = (n + 1) * (rho * g * h0 * u0 / 4)**n * icepack.rate_factor(T)
        ux = (u0**(n + 1) + q * x)**(1/(n + 1))
        u = interpolate(firedrake.as_vector((ux, 0)), V)
        h = interpolate(h0 * u0 / ux, Q)
        h_init = h.copy(deepcopy=True)
        A = interpolate(firedrake.Constant(icepack.rate_factor(T)), Q)
        a = firedrake.Function(Q)

        final_time, dt = 1.0, 1.0/12
        num_timesteps = math.ceil(final_time / dt)

        for k in range(num_timesteps):
            u = ice_shelf.diagnostic_solve(h=h, A=A, u0=u, **opts)
            h = ice_shelf.prognostic_solve(dt, h0=h, a=a, u=u)

        error.append(norm(h - h_init) / norm(h_init))
        print(delta_x[-1], error[-1])

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print(slope, intercept)
    assert slope > degree - 0.05


# Test solving the coupled diagnostic/prognostic equations for an ice stream
# and check that it doesn't explode. TODO: Manufacture a solution.
def test_ice_stream_prognostic_solve():
    from icepack.models import IceStream

    Lx, Ly = 20e3, 20e3
    h0, dh = 500.0, 100.0
    T = 254.15
    u0 = 100.0

    ice_stream = IceStream()
    opts = {"dirichlet_ids": [1, 3, 4], "tol": 1e-12}

    N = 32
    mesh = firedrake.RectangleMesh(N, N, Lx, Ly)
    x, y = firedrake.SpatialCoordinate(mesh)

    degree = 2
    V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
    Q = firedrake.FunctionSpace(mesh, 'CG', degree)

    height_above_flotation = 10.0
    d = -rho_ice / rho_water * (h0 - dh) + height_above_flotation
    rho = rho_ice - rho_water * d**2 / (h0 - dh)**2

    Z = icepack.rate_factor(T) * (rho * g * h0 / 4)**n
    q = 1 - (1 - (dh/h0) * (x/Lx))**(n + 1)
    ux = u0 + Z * q * Lx * (h0/dh) / (n + 1)
    u = interpolate(firedrake.as_vector((ux, 0)), V)

    thickness = h0 - dh * x / Lx
    beta = 1/2
    alpha = beta * rho / rho_ice * dh / Lx
    h = interpolate(h0 - dh * x / Lx, Q)
    ds = (1 + beta) * rho/rho_ice * dh
    s = interpolate(d + h0 - dh + ds * (1 - x / Lx), Q)
    b = interpolate(s - h, Q)

    from icepack.constants import weertman_sliding_law as m
    C = interpolate(alpha * (rho_ice * g * thickness) * ux**(-1/m), Q)
    A = interpolate(firedrake.Constant(icepack.rate_factor(T)), Q)

    final_time, dt = 1.0, 1.0/12
    num_timesteps = math.ceil(final_time / dt)

    a = firedrake.Function(Q)
    a = (ice_stream.prognostic_solve(dt, h0=h, a=a, u=u) - h) / dt

    for k in range(num_timesteps):
        u = ice_stream.diagnostic_solve(u0=u, h=h, s=s, C=C, A=A, **opts)
        h = ice_stream.prognostic_solve(dt, h0=h, a=a, u=u)
        s = ice_stream.compute_surface(h=h, b=b)

    assert icepack.norm(h, norm_type='Linfty') < np.inf

