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

import pytest
import numpy as np
import firedrake
from firedrake import assemble, inner, as_vector, Constant, dx, ds_t, ds_b
import icepack
from icepack.constants import (
    year,
    thermal_diffusivity as α,
    melting_temperature as Tm
)

# Using the same mesh and data for every test case.
Nx, Ny = 32, 32
Lx, Ly = 20e3, 20e3
mesh2d = firedrake.RectangleMesh(Nx, Ny, Lx, Ly)
mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)

Q = firedrake.FunctionSpace(
    mesh, family='CG', degree=2, vfamily='GL', vdegree=4
)

Q_c = firedrake.FunctionSpace(
    mesh, family='CG', degree=2, vfamily='R', vdegree=0
)

V = firedrake.VectorFunctionSpace(
    mesh, dim=2, family='CG', degree=2, vfamily='GL', vdegree=4
)

W = firedrake.FunctionSpace(
    mesh, family='DG', degree=1, vfamily='GL', vdegree=5
)

x, y, ζ = firedrake.SpatialCoordinate(mesh)

# The test glacier slopes down and thins out toward the terminus
h0, dh = 500.0, 100.0
h = firedrake.interpolate(h0 - dh * x / Lx, Q_c)

s0, ds = 500.0, 50.0
s = firedrake.interpolate(s0 - ds * x / Lx, Q_c)


# The energy density at the surface (MPa / m^3) and heat flux (MPa / m^2 / yr)
# at the ice bed
E_surface = 480
q_bed = 50e-3 * year * 1e-6


@pytest.mark.parametrize('params', [{'ksp_type': 'cg', 'pc_type': 'ilu'}, None])
def test_diffusion(params):
    E_true = firedrake.interpolate(E_surface + q_bed / α * h * (1 - ζ), Q)
    E = firedrake.interpolate(Constant(480), Q)

    # Subclass the heat transport model and turn off advection so that we can
    # test diffusion by itself
    class DiffusionTransportModel(icepack.models.HeatTransport3D):
        def __init__(self):
            super(DiffusionTransportModel, self).__init__()

        def advective_flux(self, **kwargs):
            E = kwargs['energy']
            h = kwargs['thickness']
            Q = E.function_space()
            ψ = firedrake.TestFunction(Q)
            return Constant(0) * ψ * h * dx

    model = DiffusionTransportModel()
    solver = icepack.solvers.HeatTransportSolver(
        model, solver_parameters=params
    )

    dt = 250.0
    final_time = 6000
    num_steps = int(final_time / dt) + 1
    for step in range(num_steps):
        E = solver.solve(
            dt,
            energy=E,
            thickness=h,
            energy_surface=Constant(E_surface),
            heat=Constant(0),
            heat_bed=Constant(q_bed)
        )

    assert assemble((E - E_true)**2 * ds_t) / assemble(E_true**2 * ds_t) < 1e-3
    assert assemble((E - E_true)**2 * ds_b) / assemble(E_true**2 * ds_b) < 1e-3


def test_advection():
    E_initial = firedrake.interpolate(E_surface + q_bed / α * h * (1 - ζ), Q)
    E = E_initial.copy(deepcopy=True)

    # Subclass the heat transport model and turn off diffusion so that we can
    # test advection by itself
    class AdvectionTransportModel(icepack.models.HeatTransport3D):
        def __init__(self):
            super(AdvectionTransportModel, self).__init__()

        def diffusive_flux(self, **kwargs):
            E = kwargs['energy']
            h = kwargs['thickness']
            Q = E.function_space()
            ψ = firedrake.TestFunction(Q)
            return Constant(0) * ψ * h * dx

    model = AdvectionTransportModel()
    solver = icepack.solvers.HeatTransportSolver(model)

    u0 = 100.0
    du = 100.0
    u_expr = as_vector((u0 + du * x / Lx, 0))
    u = firedrake.interpolate(u_expr, V)
    w = firedrake.interpolate((-du / Lx + dh / Lx / h * u[0]) * ζ, W)

    dt = 10.0
    final_time = Lx / u0
    num_steps = int(final_time / dt) + 1
    for step in range(num_steps):
        E = solver.solve(
            dt,
            energy=E,
            velocity=u,
            vertical_velocity=w,
            thickness=h,
            surface=s,
            heat=Constant(0),
            heat_bed=Constant(q_bed),
            energy_inflow=E_initial,
            energy_surface=Constant(E_surface)
        )

    error_surface = assemble((E - E_surface)**2 * ds_t)
    assert error_surface / assemble(E_surface**2 * ds_t(mesh)) < 1e-2
    error_bed = assemble((E - E_initial)**2 * ds_b)
    assert error_bed / assemble(E_initial**2 * ds_b(mesh)) < 1e-2


def test_advection_diffusion():
    E_initial = firedrake.interpolate(E_surface + q_bed / α * h * (1 - ζ), Q)
    E = E_initial.copy(deepcopy=True)

    u0 = 100.0
    du = 100.0
    u_expr = as_vector((u0 + du * x / Lx, 0))
    u = firedrake.interpolate(u_expr, V)
    w = firedrake.interpolate((-du / Lx + dh / Lx / h * u[0]) * ζ, W)

    dt = 10.0
    final_time = Lx / u0
    num_steps = int(final_time / dt) + 1
    model = icepack.models.HeatTransport3D()
    solver = icepack.solvers.HeatTransportSolver(model)
    for step in range(num_steps):
        E = solver.solve(
            dt,
            energy=E,
            velocity=u,
            vertical_velocity=w,
            thickness=h,
            surface=s,
            heat=Constant(0),
            heat_bed=Constant(q_bed),
            energy_inflow=E_initial,
            energy_surface=Constant(E_surface)
        )

    rms = np.sqrt(assemble(E**2 * h * dx) / assemble(h * dx))
    assert (E_surface - 5 < rms) and (rms < E_surface + 5 + q_bed / α * h0)


def test_converting_fields():
    δT = 5.0
    T_surface = Tm - δT
    T_expr = firedrake.min_value(Tm, T_surface + 2 * δT * (1 - ζ))
    f_expr = firedrake.max_value(0, 0.0033 * (1 - 2 * ζ))

    model = icepack.models.HeatTransport3D()
    E = firedrake.project(model.energy_density(T_expr, f_expr), Q)
    f = firedrake.project(model.meltwater_fraction(E), Q)
    T = firedrake.project(model.temperature(E), Q)

    avg_meltwater = firedrake.assemble(f * ds_b) / (Lx * Ly)
    assert avg_meltwater > 0

    avg_temp = firedrake.assemble(T * h * dx) / firedrake.assemble(h * dx)
    assert (avg_temp > T_surface) and (avg_temp < Tm)


def test_strain_heating():
    E_initial = firedrake.interpolate(E_surface + q_bed / α * h * (1 - ζ), Q)

    u0 = 100.0
    du = 100.0
    u_expr = as_vector((u0 + du * x / Lx, 0))
    u = firedrake.interpolate(u_expr, V)
    w = firedrake.interpolate((-du / Lx + dh / Lx / h * u[0]) * ζ, W)

    E_q = E_initial.copy(deepcopy=True)
    E_0 = E_initial.copy(deepcopy=True)

    from icepack.models.hybrid import (
        horizontal_strain,
        vertical_strain,
        stresses
    )
    model = icepack.models.HeatTransport3D()
    T = model.temperature(E_q)
    A = icepack.rate_factor(T)
    ε_x, ε_z = horizontal_strain(u, s, h), vertical_strain(u, h)
    τ_x, τ_z = stresses(ε_x, ε_z, A)
    q = firedrake.project(inner(τ_x, ε_x) + inner(τ_z, ε_z), Q)

    fields = {
        'velocity': u,
        'vertical_velocity': w,
        'thickness': h,
        'surface': s,
        'heat_bed': Constant(q_bed),
        'energy_inflow': E_initial,
        'energy_surface': Constant(E_surface)
    }

    dt = 10.0
    final_time = Lx / u0
    num_steps = int(final_time / dt) + 1
    solver_strain = icepack.solvers.HeatTransportSolver(model)
    solver_no_strain = icepack.solvers.HeatTransportSolver(model)
    for step in range(num_steps):
        E_q = solver_strain.solve(dt, energy=E_q, heat=q, **fields)
        E_0 = solver_no_strain.solve(dt, energy=E_0, heat=Constant(0), **fields)

    assert assemble(E_q * h * dx) > assemble(E_0 * h * dx)
