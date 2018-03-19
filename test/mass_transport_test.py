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
from firedrake import RectangleMesh, FunctionSpace, VectorFunctionSpace
import icepack
from icepack.models.mass_transport import MassTransport

# Test solving the mass transport equations with a constant velocity field
# and check that the solutions converge to the exact solution obtained from
# the method of characteristics.
def test_mass_transport_solver_convergence():
    L, W = 1.0, 1.0
    u0 = 1.0
    h_in, dh = 1.0, 0.2

    def thickness_initial(x):
        return h_in - dh * x[0] / L

    def thickness_exact(x, t):
        z = (x[0] - u0 * t, x[1])
        return h_in if z[0] < 0 else thickness_initial(z)

    delta_x, error = [], []
    mass_transport = MassTransport()
    norm = lambda v: icepack.norm(v, norm_type='L1')
    for N in range(16, 97, 4):
        mesh = RectangleMesh(N, N, L, W)
        delta_x.append(L / N)

        degree = 1
        V = VectorFunctionSpace(mesh, 'CG', degree)
        Q = FunctionSpace(mesh, 'CG', degree)

        h = icepack.interpolate(thickness_initial, Q)
        a = icepack.interpolate(lambda x: 0.0, Q)
        u = icepack.interpolate(lambda x: (u0, 0.0), V)
        T = 0.5
        num_timesteps = math.ceil(0.5 * N * u0 * T / L)
        dt = T / num_timesteps

        for k in range(num_timesteps):
            h = mass_transport.solve(dt, h0=h, a=a, u=u)

        h_exact = icepack.interpolate(lambda x: thickness_exact(x, T), Q)
        error.append(norm(h - h_exact) / norm(h_exact))

        print(delta_x[-1], error[-1])

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print(slope, intercept)


from icepack.constants import rho_ice, rho_water, gravity, glen_flow_law as n

# Test solving the coupled diagnostic/prognostic equations for an ice shelf
# with thickness and velocity fields that are exactly insteady state.
def test_ice_shelf_coupled_diagnostic_prognostic_solver():
    from icepack.models import IceShelf
    rho = rho_ice * (1 - rho_ice/rho_water)

    L, W = 20.0e3, 20.0e3
    h0, dh = 500.0, 100.0
    u0 = 100.0

    T = 254.15
    def velocity(x):
        q = (n + 1) * (rho * gravity * h0 * u0 / 4)**n * icepack.rate_factor(T)
        return ((u0**(n + 1) + q * x[0])**(1/(n + 1)), 0)

    def thickness(x):
        return h0 * u0 / velocity(x)[0]

    ice_shelf = IceShelf()
    degree = 2
    opts = {'dirichlet_ids': [1, 3, 4], 'tol': 1e-12}
    norm = lambda h: icepack.norm(h, norm_type='L1')

    delta_x, error = [], []
    for N in range(16, 65, 4):
        mesh = RectangleMesh(N, N, L, W)
        delta_x.append(L / N)

        V = VectorFunctionSpace(mesh, 'CG', degree)
        Q = FunctionSpace(mesh, 'CG', degree)

        u = icepack.interpolate(velocity, V)
        h = icepack.interpolate(thickness, Q)
        h_init = h.copy(deepcopy=True)
        A = icepack.interpolate(lambda x: icepack.rate_factor(T), Q)
        a = icepack.interpolate(lambda x: 0.0, Q)

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
def test_ice_stream_coupled_diagnostic_prognostic_solve():
    from icepack.models import IceStream

    L, W = 20e3, 20e3

    ice_stream = IceStream()
    degree = 2
    opts = {"dirichlet_ids": [1, 3, 4], "tol": 1e-12}

    N = 32
    mesh = RectangleMesh(N, N, L, W)
    V = VectorFunctionSpace(mesh, 'CG', degree)
    Q = FunctionSpace(mesh, 'CG', degree)

    h0, dh = 500.0, 100.0
    def thickness(x):
        return h0 - dh * x[0] / L

    height_above_flotation = 10.0
    d = -rho_ice / rho_water * thickness((L, W/2)) + height_above_flotation
    rho = rho_ice - rho_water * d**2 / thickness((L, W/2))**2

    T = 254.15
    u0 = 100.0
    def velocity_initial(x):
        A = icepack.rate_factor(T) * (rho * gravity * h0 / 4)**n
        q = 1 - (1 - (dh/h0) * (x[0]/L))**(n + 1)
        du = A * q * L * (h0/dh) / (n + 1)
        return (u0 + du, 0.0)

    beta = 1/2
    alpha = beta * rho / rho_ice * dh / L
    def friction(x):
        from icepack.constants import weertman_sliding_law as m
        u = velocity_initial(x)[0]
        h = thickness(x)
        return alpha * (rho_ice * gravity * h) * u**(-1/m)

    ds = (1 + beta) * rho/rho_ice * dh
    def surface(x):
        return d + h0 - dh + ds * (1 - x[0] / L)

    def bed(x):
        return surface(x) - thickness(x)

    u = icepack.interpolate(velocity_initial, V)
    h = icepack.interpolate(thickness, Q)
    s = icepack.interpolate(surface, Q)
    b = icepack.interpolate(bed, Q)
    C = icepack.interpolate(friction, Q)
    A = icepack.interpolate(lambda x: icepack.rate_factor(T), Q)

    final_time, dt = 1.0, 1.0/12
    num_timesteps = math.ceil(final_time / dt)

    a = icepack.interpolate(lambda x: 0.0, Q)
    a = (ice_stream.prognostic_solve(dt, h0=h, a=a, u=u) - h) / dt

    import matplotlib.pyplot as plt

    for k in range(num_timesteps):
        u = ice_stream.diagnostic_solve(u0=u, h=h, s=s, C=C, A=A, **opts)
        h = ice_stream.prognostic_solve(dt, h0=h, a=a, u=u)
        s = ice_stream.compute_surface(h=h, b=b)

    assert icepack.norm(h, norm_type='Linfty') < np.inf

