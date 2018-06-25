# Copyright (C) 2018 by Daniel Shapero <shapero@uw.edu>
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
import numpy.random as random
import firedrake
from firedrake import inner, grad, dx, interpolate, as_vector
import icepack, icepack.models, icepack.inverse
from icepack.constants import gravity as g, glen_flow_law as n, \
    rho_ice as ρ_I, rho_water as ρ_W


class PoissonModel(object):
    def action(self, a, u, f, **kwargs):
        return (0.5 * inner(grad(u), grad(u)) / a - f * u) * dx

    def quadrature_degree(self, a, u, **kwargs):
        degree_a = a.ufl_element().degree()
        degree_u = u.ufl_element().degree()
        return max(degree_a + 2 * (degree_u - 1), 2 * degree_u)

    def solve(self, a, f, dirichlet_ids=[], **kwargs):
        u = firedrake.Function(f.function_space())
        W = self.action(a, u, f)
        F = firedrake.derivative(W, u)
        bc = firedrake.DirichletBC(u.function_space(), 0, dirichlet_ids)
        firedrake.solve(F == 0, u, bc)
        return u


def test_poisson_inverse():
    Nx, Ny = 32, 32
    mesh = firedrake.UnitSquareMesh(Nx, Ny)
    degree = 2
    Q = firedrake.FunctionSpace(mesh, 'CG', degree)

    x, y = firedrake.SpatialCoordinate(mesh)
    r2 = ((x - 0.5)**2 + (y - 0.5)**2)
    a_true = interpolate(1 + firedrake.exp(-4 * r2), Q)
    f = interpolate(firedrake.Constant(1), Q)

    dirichlet_ids = [1, 2, 3, 4]
    model = PoissonModel()
    u_obs = model.solve(a=a_true, f=f, dirichlet_ids=dirichlet_ids)

    a0 = interpolate(firedrake.Constant(1), Q)
    u0 = model.solve(a=a0, f=f, dirichlet_ids=dirichlet_ids)

    def callback(inverse_problem):
        E, R = inverse_problem.objective, inverse_problem.regularization
        print(firedrake.assemble(E), firedrake.assemble(R))

    L = firedrake.Constant(1e-4)
    inverse_problem = icepack.inverse.InverseProblem(
        model=model,
        method=PoissonModel.solve,
        objective=0.5 * (u0 - u_obs)**2 * dx,
        regularization=L**2/2 * inner(grad(a0), grad(a0)) * dx,
        state_name='u',
        state=u0,
        parameter_name='a',
        parameter=a0,
        parameter_bounds=(0, 4),
        barrier=1e-6,
        model_args={'f': f},
        dirichlet_ids=dirichlet_ids,
        callback=callback
    )

    assert inverse_problem.state is not None

    inverse_problem.update()
    assert icepack.norm(inverse_problem.state) > 0
    assert icepack.norm(inverse_problem.adjoint_state) > 0
    assert icepack.norm(inverse_problem.search_direction) > 0

    max_iterations = 1000
    num_iterations = inverse_problem.solve(rtol=1e-2, atol=1e-8,
                                           max_iterations=max_iterations)
    assert num_iterations < max_iterations

    a = inverse_problem.parameter
    assert icepack.norm(a - a_true)/icepack.norm(a_true) < 0.1


def test_ice_shelf_inverse():
    import icepack
    Nx, Ny = 32, 32
    Lx, Ly = 20e3, 20e3

    u0 = 100
    h0, δh = 500, 100
    T0, δT = 254.15, 5.0
    A0 = icepack.rate_factor(T0)
    δA = icepack.rate_factor(T0 + δT) - A0

    def exact_u(x):
        ρ = ρ_I * (1 - ρ_I / ρ_W)
        Z = icepack.rate_factor(T0) * (ρ * g * h0 / 4)**n
        q = 1 - (1 - (δh/h0) * (x/Lx))**(n + 1)
        δu = Z * q * Lx * (h0/δh) / (n + 1)
        return u0 + δu

    mesh = firedrake.RectangleMesh(Nx, Ny, Lx, Ly)
    degree = 2
    V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
    Q = firedrake.FunctionSpace(mesh, 'CG', degree)

    x, y = firedrake.SpatialCoordinate(mesh)
    u_initial = interpolate(as_vector((exact_u(x), 0)), V)
    A_initial = interpolate(firedrake.Constant(A0), Q)
    h = interpolate(h0 - δh * x / Lx, Q)

    ice_shelf = icepack.models.IceShelf()
    dirichlet_ids = [1, 3, 4]
    tol = 1e-12

    r = firedrake.sqrt((x/Lx - 1/2)**2 + (y/Ly - 1/2)**2)
    R = 1/4
    expr = firedrake.max_value(0, δA * (1 - (r/R)**2))
    A_true = firedrake.interpolate(A0 + expr, Q)
    u_true = ice_shelf.diagnostic_solve(h=h, A=A_true, u0=u_initial,
                                        dirichlet_ids=dirichlet_ids, tol=tol)

    area = firedrake.assemble(firedrake.Constant(1) * dx(mesh))
    def callback(inverse_problem):
        E, R = inverse_problem.objective, inverse_problem.regularization
        print(firedrake.assemble(E) / area, firedrake.assemble(R) / area)

    L = 1e-4 * Lx
    regularization = L**2/2 * inner(grad(A_initial), grad(A_initial)) * dx
    inverse_problem = icepack.inverse.InverseProblem(
        model=ice_shelf,
        method=icepack.models.IceShelf.diagnostic_solve,
        objective=0.5 * (u_initial - u_true)**2 * dx,
        regularization=regularization,
        state_name='u',
        state=u_initial,
        parameter_name='A',
        parameter=A_initial,
        parameter_bounds=(A0 / 2, 1.5 * (A0 + δA)),
        barrier=1e-3,
        model_args={'h': h, 'u0': u_initial, 'tol': tol},
        dirichlet_ids=dirichlet_ids,
        callback=callback
    )

    # Set an absolute tolerance so that we stop whenever the RMS velocity
    # errors are less than 0.1 m/yr
    area = firedrake.assemble(firedrake.Constant(1) * dx(mesh))
    atol = 0.5 * 0.01 * area

    max_iterations = 100
    iters = inverse_problem.solve(rtol=1e-2, atol=atol,
                                  max_iterations=100)
    assert iters < max_iterations


def test_ice_shelf_inverse_with_noise():
    import icepack
    Nx, Ny = 32, 32
    Lx, Ly = 20e3, 20e3

    u0 = 100
    h0, δh = 500, 100
    T0, δT = 254.15, 5.0
    A0 = icepack.rate_factor(T0)
    δA = icepack.rate_factor(T0 + δT) - A0

    def exact_u(x):
        ρ = ρ_I * (1 - ρ_I / ρ_W)
        Z = icepack.rate_factor(T0) * (ρ * g * h0 / 4)**n
        q = 1 - (1 - (δh/h0) * (x/Lx))**(n + 1)
        δu = Z * q * Lx * (h0/δh) / (n + 1)
        return u0 + δu

    mesh = firedrake.RectangleMesh(Nx, Ny, Lx, Ly)
    degree = 2
    V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)
    Q = firedrake.FunctionSpace(mesh, 'CG', degree)

    x, y = firedrake.SpatialCoordinate(mesh)
    u_initial = interpolate(as_vector((exact_u(x), 0)), V)
    A_initial = interpolate(firedrake.Constant(A0), Q)
    h = interpolate(h0 - δh * x / Lx, Q)

    ice_shelf = icepack.models.IceShelf()
    dirichlet_ids = [1, 3, 4]
    tol = 1e-12

    r = firedrake.sqrt((x/Lx - 1/2)**2 + (y/Ly - 1/2)**2)
    R = 1/4
    expr = firedrake.max_value(0, δA * (1 - (r/R)**2))
    A_true = firedrake.interpolate(A0 + expr, Q)
    u_true = ice_shelf.diagnostic_solve(h=h, A=A_true, u0=u_initial,
                                        dirichlet_ids=dirichlet_ids, tol=tol)

    # Make the noise equal to 1% of the signal
    area = firedrake.assemble(firedrake.Constant(1) * dx(mesh))
    σ = 0.01 * icepack.norm(u_true) / np.sqrt(area)
    print(σ)

    u_obs = u_true.copy(deepcopy=True)
    u_obs.dat.data[:] += σ * random.standard_normal(u_obs.dat.data_ro.shape)

    def callback(inverse_problem):
        E = inverse_problem.objective
        R = inverse_problem.regularization
        print(firedrake.assemble(E) / area, firedrake.assemble(R) / area)

    L = 5e-2 * Lx
    regularization = L**2/2 * inner(grad(A_initial), grad(A_initial)) * dx
    inverse_problem = icepack.inverse.InverseProblem(
        model=ice_shelf,
        method=icepack.models.IceShelf.diagnostic_solve,
        objective=0.5 * ((u_initial - u_obs)/σ)**2 * dx,
        regularization=regularization,
        state_name='u',
        state=u_initial,
        parameter_name='A',
        parameter=A_initial,
        parameter_bounds=(A0 / 2, 1.5 * (A0 + δA)),
        barrier=1e-3,
        model_args={'h': h, 'u0': u_initial, 'tol': tol},
        dirichlet_ids=dirichlet_ids,
        callback=callback
    )

    # Set an absolute tolerance so that we stop whenever the RMS velocity
    # errors are less than 0.1 m/yr
    max_iterations = 100
    iters = inverse_problem.solve(rtol=2.5e-3, atol=0.0, max_iterations=100)
    assert iters < max_iterations

    import matplotlib.pyplot as plt
    import icepack.plot
    fig, (ax1, ax2) = plt.subplots(2)
    icepack.plot.tricontourf(inverse_problem.state, axes=ax1)
    icepack.plot.tricontourf(inverse_problem.parameter, axes=ax2)
    plt.show(fig)

