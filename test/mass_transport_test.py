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

import pytest
import numpy as np
import firedrake
from firedrake import interpolate, assemble, dx
import icepack

def norm(v):
    return icepack.norm(v, norm_type='L1')


# Test solving the mass transport equations with a constant velocity field
# and check that the solutions converge to the exact solution obtained from
# the method of characteristics.
@pytest.mark.parametrize('solver_type', ['implicit-euler', 'lax-wendroff'])
def test_mass_transport_solver_convergence(solver_type):
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
        solver = icepack.solvers.FlowSolver(
            model, prognostic_solver_type=solver_type
        )

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
        error.append(norm(h - h_exact) / norm(h_exact))
        print(delta_x[-1], error[-1])

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print('log(error) ~= {:g} * log(dx) + {:g}'.format(slope, intercept))
    assert slope > degree - 0.1


from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    glen_flow_law as n,
    weertman_sliding_law as m,
    gravity as g
)


# Test solving the coupled diagnostic/prognostic equations for an ice shelf
# with thickness and velocity fields that are exactly insteady state.
@pytest.mark.parametrize('solver_type', ['implicit-euler', 'lax-wendroff'])
def test_ice_shelf_prognostic_solver(solver_type):
    ρ = ρ_I * (1 - ρ_I / ρ_W)

    Lx, Ly = 20.0e3, 20.0e3
    h0 = 500.0
    u0 = 100.0
    T = 254.15

    model = icepack.models.IceShelf()
    opts = {
        'dirichlet_ids': [1],
        'side_wall_ids': [3, 4],
        'prognostic_solver_type': solver_type
    }

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

        final_time, dt = 1.0, 1.0/12
        num_timesteps = int(final_time / dt)

        for k in range(num_timesteps):
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

        error.append(norm(h - h_initial) / norm(h_initial))
        print(delta_x[-1], error[-1])

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print('log(error) ~= {:g} * log(dx) + {:g}'.format(slope, intercept))
    assert slope > degree - 0.05


# Test solving the shallow ice approximation forward in time with no
# accumulation and outflow and check that the total ice volume is conserved
def test_shallow_ice_prognostic_solve():
    R = firedrake.Constant(500e3)
    num_refinements = 4
    mesh = firedrake.UnitDiskMesh(num_refinements)
    mesh.coordinates.dat.data[:] *= float(R)
    T = firedrake.Constant(254.15)
    A = icepack.rate_factor(T)

    Q = firedrake.FunctionSpace(mesh, family='CG', degree=2)
    V = firedrake.VectorFunctionSpace(mesh, family='CG', degree=2)

    x, y = firedrake.SpatialCoordinate(mesh)
    r = firedrake.sqrt(x**2 + y**2)

    β = firedrake.Constant(0.5)
    h_divide = firedrake.Constant(4e3)
    h_expr = h_divide * firedrake.max_value(0, 1 - (r / (β * R))**2)
    h_0 = interpolate(h_expr, Q)
    h = h_0.copy(deepcopy=True)
    u = firedrake.Function(V)

    b = firedrake.Constant(0.)
    s = interpolate(b + h, Q)
    a = firedrake.Constant(0.)

    model = icepack.models.ShallowIce()
    solver = icepack.solvers.FlowSolver(model)

    final_time = 100.
    dt = 1.
    num_steps = int(final_time / dt)

    for step in range(num_steps):
        u = solver.diagnostic_solve(
            velocity=u, thickness=h, surface=s, fluidity=A
        )

        h = solver.prognostic_solve(
            dt, thickness=h, velocity=u, accumulation=a
        )

        h.interpolate(firedrake.max_value(0, h))
        s.assign(b + h)

    error = abs(assemble(h * dx) / assemble(h_0 * dx) - 1)
    assert error < 1 / 2 ** (num_refinements + 1)


# Test solving the coupled diagnostic/prognostic equations for an ice stream
# and check that it doesn't explode. TODO: Manufacture a solution.
def test_ice_stream_prognostic_solve():
    Lx, Ly = 20e3, 20e3
    h0, dh = 500.0, 100.0
    T = 254.15
    u0 = 100.0

    model = icepack.models.IceStream()
    opts = {'dirichlet_ids': [1], 'side_wall_ids': [3, 4]}

    N = 32
    mesh = firedrake.RectangleMesh(N, N, Lx, Ly)

    V = firedrake.VectorFunctionSpace(mesh, family='CG', degree=2)
    Q = firedrake.FunctionSpace(mesh, family='CG', degree=2)
    solver = icepack.solvers.FlowSolver(model, **opts)

    x, y = firedrake.SpatialCoordinate(mesh)
    height_above_flotation = 10.0
    d = -ρ_I / ρ_W * (h0 - dh) + height_above_flotation
    ρ = ρ_I - ρ_W * d**2 / (h0 - dh)**2

    Z = icepack.rate_factor(T) * (ρ * g * h0 / 4)**n
    q = 1 - (1 - (dh/h0) * (x/Lx))**(n + 1)
    ux = u0 + Z * q * Lx * (h0/dh) / (n + 1)
    u0 = interpolate(firedrake.as_vector((ux, 0)), V)

    thickness = h0 - dh * x / Lx
    β = 1/2
    α = β * ρ / ρ_I * dh / Lx
    h = interpolate(h0 - dh * x / Lx, Q)
    h_inflow = h.copy(deepcopy=True)
    ds = (1 + β) * ρ / ρ_I * dh
    s = interpolate(d + h0 - dh + ds * (1 - x / Lx), Q)
    b = interpolate(s - h, Q)

    C = interpolate(α * (ρ_I * g * thickness) * ux**(-1/m), Q)
    A = firedrake.Constant(icepack.rate_factor(T))

    final_time, dt = 1.0, 1.0/12
    num_timesteps = int(final_time / dt)

    u = solver.diagnostic_solve(
        velocity=u0, thickness=h, surface=s, friction=C, fluidity=A
    )

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
            friction=C,
            fluidity=A
        )

    assert icepack.norm(h, norm_type='Linfty') < np.inf


# Test solving the coupled diagnostic/prognostic equations for the hybrid flow
# model and check that it doesn't explode.
@pytest.mark.parametrize('solver_type', ['implicit-euler', 'lax-wendroff'])
def test_hybrid_prognostic_solve(solver_type):
    Lx, Ly = 20e3, 20e3
    h0, dh = 500.0, 100.0
    T = 254.15
    u_in = 100.0

    model = icepack.models.HybridModel()
    opts = {
        'dirichlet_ids': [1],
        'side_wall_ids': [3, 4],
        'prognostic_solver_type': solver_type
    }

    Nx, Ny = 32, 32
    mesh2d = firedrake.RectangleMesh(Nx, Ny, Lx, Ly)
    mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)

    V = firedrake.VectorFunctionSpace(
        mesh, 'CG', 2, vfamily='GL', vdegree=1, dim=2
    )
    Q = firedrake.FunctionSpace(mesh, 'CG', 2, vfamily='DG', vdegree=0)

    x, y, ζ = firedrake.SpatialCoordinate(mesh)
    height_above_flotation = 10.0
    d = -ρ_I / ρ_W * (h0 - dh) + height_above_flotation
    ρ = ρ_I - ρ_W * d**2 / (h0 - dh)**2

    Z = icepack.rate_factor(T) * (ρ * g * h0 / 4)**n
    q = 1 - (1 - (dh/h0) * (x/Lx))**(n + 1)
    ux = u_in + Z * q * Lx * (h0/dh) / (n + 1)
    u0 = interpolate(firedrake.as_vector((ux, 0)), V)

    thickness = h0 - dh * x / Lx
    β = 1/2
    α = β * ρ / ρ_I * dh / Lx
    h = interpolate(h0 - dh * x / Lx, Q)
    h_inflow = h.copy(deepcopy=True)
    ds = (1 + β) * ρ / ρ_I * dh
    s = interpolate(d + h0 - dh + ds * (1 - x / Lx), Q)
    b = interpolate(s - h, Q)

    C = interpolate(α * (ρ_I * g * thickness) * ux**(-1/m), Q)
    A = firedrake.Constant(icepack.rate_factor(T))

    final_time, dt = 1.0, 1.0/12
    num_timesteps = int(final_time / dt)

    solver = icepack.solvers.FlowSolver(model, **opts)
    u = solver.diagnostic_solve(
        velocity=u0, thickness=h, surface=s, fluidity=A, friction=C
    )

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
