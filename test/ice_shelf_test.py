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
import firedrake
from firedrake import interpolate, as_vector
import icepack
from icepack import norm
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g, glen_flow_law as n
)

# The domain is a 20km x 20km square, with ice flowing in from the left.
Lx, Ly = 20.0e3, 20.0e3

# The inflow velocity is 100 m/year; the ice shelf decreases from 500m thick
# to 100m; and the temperature is a constant -19C.
u0 = 100.0
h0, dh = 500.0, 100.0
T = 254.15


# This is an exact solution for the velocity of a floating ice shelf with
# constant temperature and linearly decreasing thickness. See Greve and
# Blatter for the derivation.
def exact_u(x):
    ρ = ρ_I * (1 - ρ_I / ρ_W)
    Z = icepack.rate_factor(T) * (ρ * g * h0 / 4)**n
    q = 1 - (1 - (dh/h0) * (x/Lx))**(n + 1)
    du = Z * q * Lx * (h0/dh) / (n + 1)
    return u0 + du


# We'll use the same perturbation to `u` throughout these tests.
def perturb_u(x, y):
    px, py = x/Lx, y/Ly
    q = 16 * px * (1 - px) * py * (1 - py)
    return 60 * q * (px - 0.5)


# Check that the diagnostic solver converges with the expected rate as the
# mesh is refined using an exact solution of the ice shelf model.
def test_diagnostic_solver_convergence():
    # Create an ice shelf model
    model = icepack.models.IceShelf()
    opts = {'dirichlet_ids': [1], 'side_wall_ids': [3, 4]}

    # Solve the ice shelf model for successively higher mesh resolution
    for degree in range(1, 4):
        delta_x, error = [], []
        for N in range(16, 97 - 32 * (degree - 1), 4):
            mesh = firedrake.RectangleMesh(N, N, Lx, Ly)
            x, y = firedrake.SpatialCoordinate(mesh)

            V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
            Q = firedrake.FunctionSpace(mesh, 'CG', degree)

            u_exact = interpolate(as_vector((exact_u(x), 0)), V)
            u_guess = interpolate(u_exact + as_vector((perturb_u(x, y), 0)), V)

            h = interpolate(h0 - dh * x / Lx, Q)
            A = interpolate(firedrake.Constant(icepack.rate_factor(T)), Q)

            solver = icepack.solvers.FlowSolver(model, **opts)
            u = solver.diagnostic_solve(
                velocity=u_guess,
                thickness=h,
                fluidity=A
            )
            error.append(norm(u_exact - u, 'H1') / norm(u_exact, 'H1'))
            delta_x.append(Lx / N)

        # Fit the error curve and check that the convergence rate is what we
        # expect
        log_delta_x = np.log2(np.array(delta_x))
        log_error = np.log2(np.array(error))
        slope, intercept = np.polyfit(log_delta_x, log_error, 1)

        print('log(error) ~= {:g} * log(dx) + {:g}'.format(slope, intercept))
        assert slope > degree + 0.8


# Check that the diagnostic solver converges with the expected rate as the
# mesh is refined when we use an alternative parameterization of the model.
def test_diagnostic_solver_parameterization():
    # Define a new viscosity functional, parameterized in terms of the
    # rheology `B` instead of the fluidity `A`
    from firedrake import inner, grad, sym, tr as trace, Identity, sqrt

    def M(ε, B):
        I = Identity(2)
        tr_ε = trace(ε)
        ε_e = sqrt((inner(ε, ε) + tr_ε**2) / 2)
        μ = 0.5 * B * ε_e**(1/n - 1)
        return 2 * μ * (ε + tr_ε * I)

    def ε(u):
        return sym(grad(u))

    def viscosity(**kwargs):
        u = kwargs['velocity']
        h = kwargs['thickness']
        B = kwargs['rheology']
        return n/(n + 1) * h * inner(M(ε(u), B), ε(u))

    # Make a model object with our new viscosity functional
    model = icepack.models.IceShelf(viscosity=viscosity)
    opts = {'dirichlet_ids': [1, 3, 4]}

    # Same as before
    delta_x, error = [], []
    for N in range(16, 65, 4):
        mesh = firedrake.RectangleMesh(N, N, Lx, Ly)
        x, y = firedrake.SpatialCoordinate(mesh)

        degree = 2
        V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
        Q = firedrake.FunctionSpace(mesh, 'CG', degree)

        u_exact = interpolate(as_vector((exact_u(x), 0)), V)
        u_guess = interpolate(as_vector((exact_u(x) + perturb_u(x, y), 0)), V)
        h = interpolate(h0 - dh * x / Lx, Q)
        B = interpolate(firedrake.Constant(icepack.rate_factor(T)**(-1/n)), Q)

        solver = icepack.solvers.FlowSolver(model, **opts)
        u = solver.diagnostic_solve(
            velocity=u_guess,
            thickness=h,
            rheology=B
        )
        error.append(norm(u_exact - u, 'H1') / norm(u_exact, 'H1'))
        delta_x.append(Lx / N)

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print('log(error) ~= {:g} * log(dx) + {:g}'.format(slope, intercept))
    assert slope > degree - 0.05


# Check that the diagnostic solver gives a sensible result when we add friction
# at the side walls. There is probably no analytical solution for this so all
# we have is a sanity test.
def test_diagnostic_solver_side_friction():
    model = icepack.models.IceShelf()
    opts = {'dirichlet_ids': [1], 'side_wall_ids': [3, 4]}

    mesh = firedrake.RectangleMesh(32, 32, Lx, Ly)
    degree = 2
    V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
    Q = firedrake.FunctionSpace(mesh, 'CG', degree)

    x, y = firedrake.SpatialCoordinate(mesh)
    u_initial = interpolate(as_vector((exact_u(x), 0)), V)
    h = interpolate(h0 - dh * x / Lx, Q)
    A = interpolate(firedrake.Constant(icepack.rate_factor(T)), Q)

    # Choose the side wall friction coefficient so that, assuming the ice is
    # sliding at the maximum speed for the solution without friction, the
    # stress is 10 kPa.
    from icepack.constants import weertman_sliding_law as m
    τ = 0.01
    u_max = norm(u_initial, 'Linfty')
    Cs = firedrake.Constant(τ * u_max**(-1/m))

    solver = icepack.solvers.FlowSolver(model, **opts)
    fields = {
        'velocity': u_initial,
        'thickness': h,
        'fluidity': A,
        'side_friction': Cs
    }
    u = solver.diagnostic_solve(**fields)

    assert icepack.norm(u) < icepack.norm(u_initial)


# Try using different options for the diagnostic solve
def test_diagnostic_solver_options():
    model = icepack.models.IceShelf()
    nx, ny = 32, 32
    mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly)

    degree = 2
    V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
    Q = firedrake.FunctionSpace(mesh, 'CG', degree)

    x, y = firedrake.SpatialCoordinate(mesh)
    u_initial = interpolate(as_vector((exact_u(x), 0)), V)
    h = interpolate(h0 - dh * x / Lx, Q)
    A = interpolate(firedrake.Constant(icepack.rate_factor(T)), Q)

    from icepack.constants import weertman_sliding_law as m
    τ = 0.01
    u_max = norm(u_initial, 'Linfty')
    Cs = firedrake.Constant(τ * u_max**(-1/m))

    opts = {'dirichlet_ids': [1], 'side_wall_ids': [3, 4]}

    from icepack.utilities import default_solver_parameters
    direct_solver = icepack.solvers.FlowSolver(
        model, **opts, diagnostic_solver_parameters=default_solver_parameters
    )

    iterative_parameters = {'ksp_type': 'cg', 'pc_type': 'ilu'}
    iterative_solver = icepack.solvers.FlowSolver(
        model, **opts, diagnostic_solver_parameters=iterative_parameters
    )

    fields = {
        'velocity': u_initial,
        'thickness': h,
        'fluidity': A,
        'side_friction': Cs
    }

    u_direct = direct_solver.diagnostic_solve(**fields)
    u_iterative = iterative_solver.diagnostic_solve(**fields)

    tol = 1 / nx ** degree
    assert norm(u_direct - u_iterative, 'H1') < tol * norm(u_direct, 'H1')


# Test solving the mass transport equations with a constant velocity field
# and check that the solutions converge to the exact solution obtained from
# the method of characteristics.
def test_mass_transport_solver_convergence():
    Lx, Ly = 1.0, 1.0
    u0 = 1.0
    h_in, dh = 1.0, 0.2

    delta_x, error = [], []
    model = icepack.models.IceShelf()
    for N in range(24, 97, 4):
        delta_x.append(Lx / N)

        mesh = firedrake.RectangleMesh(N, N, Lx, Ly)
        x, y = firedrake.SpatialCoordinate(mesh)

        degree = 1
        V = firedrake.VectorFunctionSpace(mesh, family='CG', degree=degree)
        Q = firedrake.FunctionSpace(mesh, family='CG', degree=degree)
        solver = icepack.solvers.FlowSolver(model)

        h0 = interpolate(h_in - dh * x / Lx, Q)
        a = firedrake.Function(Q)
        u = interpolate(firedrake.as_vector((u0, 0)), V)
        T = 0.5
        δx = 1.0 / N
        δt = δx / u0
        num_timesteps = int(T / δt)

        h = h0.copy(deepcopy=True)
        for step in range(num_timesteps):
            h = solver.prognostic_solve(
                δt,
                thickness=h,
                velocity=u,
                accumulation=a,
                thickness_inflow=h0
            )

        z = x - u0 * num_timesteps * δt
        h_exact = interpolate(h_in - dh/Lx * firedrake.max_value(0, z), Q)
        error.append(norm(h - h_exact, 'L1') / norm(h_exact, 'L1'))

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print('log(error) ~= {:g} * log(dx) + {:g}'.format(slope, intercept))
    assert slope > degree - 0.1


# Test solving the coupled diagnostic/prognostic equations for an ice shelf
# with thickness and velocity fields that are exactly insteady state.
def test_ice_shelf_prognostic_solver():
    ρ = ρ_I * (1 - ρ_I / ρ_W)

    Lx, Ly = 20.0e3, 20.0e3
    h0 = 500.0
    u0 = 100.0
    T = 254.15

    model = icepack.models.IceShelf()
    opts = {'dirichlet_ids': [1], 'side_wall_ids': [3, 4]}

    delta_x, error = [], []
    for N in range(16, 65, 4):
        delta_x.append(Lx / N)

        mesh = firedrake.RectangleMesh(N, N, Lx, Ly)
        x, y = firedrake.SpatialCoordinate(mesh)

        degree = 2
        V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
        Q = firedrake.FunctionSpace(mesh, 'CG', degree)

        q = (n + 1) * (ρ * g * h0 * u0 / 4)**n * icepack.rate_factor(T)
        ux = (u0**(n + 1) + q * x)**(1/(n + 1))

        h = interpolate(h0 * u0 / ux, Q)
        h_initial = h.copy(deepcopy=True)

        A = firedrake.Constant(icepack.rate_factor(T))
        a = firedrake.Constant(0)

        solver = icepack.solvers.FlowSolver(model, **opts)
        u_guess = interpolate(firedrake.as_vector((ux, 0)), V)
        u = solver.diagnostic_solve(
            velocity=u_guess, thickness=h, fluidity=A
        )

        final_time = 1.0
        timestep = 2 / N
        num_timesteps = int(final_time / timestep)
        dt = final_time / num_timesteps

        for step in range(num_timesteps):
            h = solver.prognostic_solve(
                dt,
                thickness=h,
                velocity=u,
                accumulation=a,
                thickness_inflow=h_initial
            )

            u = solver.diagnostic_solve(
                velocity=u,
                thickness=h,
                fluidity=A
            )

        error.append(norm(h - h_initial, 'L1') / norm(h_initial, 'L1'))

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print('log(error) ~= {:g} * log(dx) + {:g}'.format(slope, intercept))
    assert slope > degree - 0.05
