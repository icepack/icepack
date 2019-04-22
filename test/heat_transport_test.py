# Copyright (C) 2019 by Daniel Shapero <shapero@uw.edu>
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
from firedrake import assemble, inner, as_vector, Constant, dx, ds_t, ds_b
import icepack.models
from icepack.constants import (year, thermal_diffusivity as α,
                               melting_temperature as Tm)

# Using the same mesh and data for every test case.
Nx, Ny = 32, 32
Lx, Ly = 20e3, 20e3
mesh2d = firedrake.RectangleMesh(Nx, Ny, Lx, Ly)
mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)

Q = firedrake.FunctionSpace(mesh, family='CG', degree=2,
                            vfamily='GL', vdegree=4)
V = firedrake.VectorFunctionSpace(mesh, dim=2, family='CG', degree=2,
                                  vfamily='GL', vdegree=4)
W = firedrake.FunctionSpace(mesh, family='DG', degree=1,
                            vfamily='GL', vdegree=5)

x, y, ζ = firedrake.SpatialCoordinate(mesh)

# The test glacier slopes down and thins out toward the terminus
h0, dh = 500.0, 100.0
h = h0 - dh * x / Lx

s0, ds = 500.0, 50.0
s = s0 - ds * x / Lx


# The energy density at the surface (MPa / m^3) and heat flux (MPa / m^2 / yr)
# at the ice bed
E_surface = 480
q_bed = 50e-3 * year * 1e-6


def test_diffusion():
    E_true = firedrake.interpolate(E_surface + q_bed / α * h * (1 - ζ), Q)
    E = firedrake.interpolate(Constant(480), Q)

    dt = 250.0
    final_time = 6000
    num_steps = int(final_time / dt) + 1
    model = icepack.models.HeatTransport3D()
    for step in range(num_steps):
        model._diffuse(dt, E=E, h=h, E_surface=Constant(E_surface),
                       q=Constant(0), q_bed=Constant(q_bed))

    assert assemble((E - E_true)**2 * ds_t) / assemble(E_true**2 * ds_t) < 1e-3
    assert assemble((E - E_true)**2 * ds_b) / assemble(E_true**2 * ds_b) < 1e-3


def test_advection():
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
    for step in range(num_steps):
        model._advect(dt, E=E, u=u, w=w, h=h, s=s,
                      E_inflow=E_initial, E_surface=Constant(E_surface))

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
    for step in range(num_steps):
        E = model.solve(dt, E0=E, u=u, w=w, h=h, s=s,
                        q=Constant(0), q_bed=q_bed,
                        E_inflow=E_initial, E_surface=Constant(E_surface))

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

    model = icepack.models.HeatTransport3D()
    from icepack.models.hybrid import (horizontal_strain, vertical_strain,
                                       stresses)
    T = model.temperature(E_q)
    A = icepack.rate_factor(T)
    ε_x, ε_z = horizontal_strain(u, s, h), vertical_strain(u, h)
    τ_x, τ_z = stresses(ε_x, ε_z, A)
    q = inner(τ_x, ε_x) + inner(τ_z, ε_z)

    kwargs = {'u': u, 'w': w, 'h': h, 's': s, 'q_bed': q_bed,
              'E_inflow': E_initial, 'E_surface': Constant(E_surface)}

    dt = 10.0
    final_time = Lx / u0
    num_steps = int(final_time / dt) + 1
    for step in range(num_steps):
        E_q.assign(model.solve(dt, E0=E_q, q=q, **kwargs))
        E_0.assign(model.solve(dt, E0=E_0, q=Constant(0), **kwargs))

    assert assemble(E_q * h * dx) > assemble(E_0 * h * dx)
