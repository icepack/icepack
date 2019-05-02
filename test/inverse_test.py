# Copyright (C) 2018-2019 by Daniel Shapero <shapero@uw.edu>
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
import numpy.random as random
import firedrake
from firedrake import inner, grad, dx, exp, interpolate, as_vector
import icepack, icepack.models
from icepack.inverse import GradientDescentSolver, BFGSSolver, \
    GaussNewtonSolver
from icepack.constants import gravity as g, glen_flow_law as n, \
    rho_ice as ρ_I, rho_water as ρ_W


class PoissonModel(object):
    def action(self, q, u, f, **kwargs):
        return (0.5 * exp(q) * inner(grad(u), grad(u)) - f * u) * dx

    def quadrature_degree(self, q, u, **kwargs):
        degree_q = q.ufl_element().degree()
        degree_u = u.ufl_element().degree()
        return max(degree_q + 2 * (degree_u - 1), 2 * degree_u)

    def solve(self, q, f, dirichlet_ids=[], **kwargs):
        u = firedrake.Function(f.function_space())
        L = self.action(q, u, f)
        F = firedrake.derivative(L, u)
        V = u.function_space()
        bc = firedrake.DirichletBC(V, firedrake.Constant(0), dirichlet_ids)
        firedrake.solve(F == 0, u, bc,
            solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
        return u


@pytest.mark.parametrize('solver_type',
    [GradientDescentSolver, BFGSSolver, GaussNewtonSolver])
def test_poisson_inverse(solver_type):
    Nx, Ny = 32, 32
    mesh = firedrake.UnitSquareMesh(Nx, Ny)
    degree = 2
    Q = firedrake.FunctionSpace(mesh, 'CG', degree)

    x, y = firedrake.SpatialCoordinate(mesh)
    q_true = interpolate(-4 * ((x - 0.5)**2 + (y - 0.5)**2), Q)
    f = interpolate(firedrake.Constant(1), Q)

    dirichlet_ids = [1, 2, 3, 4]
    model = PoissonModel()
    u_obs = model.solve(q=q_true, f=f, dirichlet_ids=dirichlet_ids)

    q0 = interpolate(firedrake.Constant(0), Q)
    u0 = model.solve(q=q0, f=f, dirichlet_ids=dirichlet_ids)

    def callback(inverse_solver):
        misfit = firedrake.assemble(inverse_solver.objective)
        regularization = firedrake.assemble(inverse_solver.regularization)
        q = inverse_solver.parameter
        error = firedrake.norm(q - q_true)
        print(misfit, regularization, error)

    L = firedrake.Constant(1e-4)
    problem = icepack.inverse.InverseProblem(
        model=model,
        method=PoissonModel.solve,
        objective=lambda u: 0.5 * (u - u_obs)**2 * dx,
        regularization=lambda q: L**2/2 * inner(grad(q), grad(q)) * dx,
        state_name='u',
        state=u0,
        parameter_name='q',
        parameter=q0,
        model_args={'f': f},
        dirichlet_ids=dirichlet_ids
    )

    solver = solver_type(problem, callback)
    assert solver.state is not None
    assert icepack.norm(solver.state) > 0
    assert icepack.norm(solver.adjoint_state) > 0
    assert icepack.norm(solver.search_direction) > 0

    max_iterations = 1000
    iterations = solver.solve(rtol=2.5e-2, atol=1e-8,
                               max_iterations=max_iterations)
    print('Number of iterations: {}'.format(iterations))

    assert iterations < max_iterations
    q = solver.parameter
    assert icepack.norm(q - q_true) < 0.25


@pytest.mark.parametrize('solver_type', [BFGSSolver, GaussNewtonSolver])
def test_ice_shelf_inverse(solver_type):
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
    q_initial = interpolate(firedrake.Constant(0), Q)
    h = interpolate(h0 - δh * x / Lx, Q)

    def viscosity(u, h, q):
        A = A0 * firedrake.exp(-q / n)
        return icepack.models.viscosity.viscosity_depth_averaged(u, h, A)

    ice_shelf = icepack.models.IceShelf(viscosity=viscosity)
    dirichlet_ids = [1, 3, 4]
    tol = 1e-12

    r = firedrake.sqrt((x/Lx - 1/2)**2 + (y/Ly - 1/2)**2)
    R = 1/4
    expr = firedrake.max_value(0, δA * (1 - (r/R)**2))
    q_true = firedrake.interpolate(-n * firedrake.ln(1 + expr / A0), Q)
    u_true = ice_shelf.diagnostic_solve(h=h, q=q_true, u0=u_initial,
                                        dirichlet_ids=dirichlet_ids, tol=tol)

    area = firedrake.assemble(firedrake.Constant(1) * dx(mesh))

    def callback(inverse_solver):
        E, R = inverse_solver.objective, inverse_solver.regularization
        misfit = firedrake.assemble(E) / area
        regularization = firedrake.assemble(R) / area
        q = inverse_solver.parameter
        error = firedrake.norm(q - q_true) / np.sqrt(area)
        print(misfit, regularization, error)

    L = 1e-4 * Lx
    problem = icepack.inverse.InverseProblem(
        model=ice_shelf,
        method=icepack.models.IceShelf.diagnostic_solve,
        objective=lambda u: 0.5 * (u - u_true)**2 * dx,
        regularization=lambda q: 0.5 * L**2 * inner(grad(q), grad(q)) * dx,
        state_name='u',
        state=u_initial,
        parameter_name='q',
        parameter=q_initial,
        model_args={'h': h, 'u0': u_initial, 'tol': tol},
        dirichlet_ids=dirichlet_ids
    )

    solver = solver_type(problem, callback)

    # Set an absolute tolerance so that we stop whenever the RMS velocity
    # errors are less than 0.1 m/yr
    area = firedrake.assemble(firedrake.Constant(1) * dx(mesh))
    atol = 0.5 * 0.01 * area
    max_iterations = 100
    iterations = solver.solve(rtol=0, atol=atol, max_iterations=max_iterations)
    print('Number of iterations: {}'.format(iterations))

    assert iterations < max_iterations
    q = solver.parameter
    assert firedrake.norm(q - q_true)/firedrake.norm(q_initial - q_true) < 1/4


@pytest.mark.parametrize('solver_type', [BFGSSolver, GaussNewtonSolver])
def test_ice_shelf_inverse_with_noise(solver_type):
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
    q_initial = interpolate(firedrake.Constant(0), Q)
    h = interpolate(h0 - δh * x / Lx, Q)

    def viscosity(u, h, q):
        A = A0 * firedrake.exp(-q / n)
        return icepack.models.viscosity.viscosity_depth_averaged(u, h, A)

    ice_shelf = icepack.models.IceShelf(viscosity=viscosity)
    dirichlet_ids = [1, 3, 4]
    tol = 1e-12

    r = firedrake.sqrt((x/Lx - 1/2)**2 + (y/Ly - 1/2)**2)
    R = 1/4
    expr = firedrake.max_value(0, δA * (1 - (r/R)**2))
    q_true = firedrake.interpolate(-n * firedrake.ln(1 + expr / A0), Q)
    u_true = ice_shelf.diagnostic_solve(h=h, q=q_true, u0=u_initial,
                                        dirichlet_ids=dirichlet_ids, tol=tol)

    # Make the noise equal to 1% of the signal
    area = firedrake.assemble(firedrake.Constant(1) * dx(mesh))
    σ = 0.001 * icepack.norm(u_true) / np.sqrt(area)
    print(σ)

    u_obs = u_true.copy(deepcopy=True)
    shape = u_obs.dat.data_ro.shape
    u_obs.dat.data[:] += σ * random.standard_normal(shape) / np.sqrt(2)

    def callback(inverse_solver):
        E, R = inverse_solver.objective, inverse_solver.regularization
        misfit = firedrake.assemble(E) / area
        regularization = firedrake.assemble(R) / area
        q = inverse_solver.parameter
        error = firedrake.norm(q - q_true) / np.sqrt(area)
        print(misfit, regularization, error, flush=True)

    L = 0.25 * Lx
    regularization = L**2/2 * inner(grad(q_initial), grad(q_initial)) * dx
    problem = icepack.inverse.InverseProblem(
        model=ice_shelf,
        method=icepack.models.IceShelf.diagnostic_solve,
        objective=lambda u: 0.5 * ((u - u_obs)/σ)**2 * dx,
        regularization=lambda q: 0.5 * L**2 * inner(grad(q), grad(q)) * dx,
        state_name='u',
        state=u_initial,
        parameter_name='q',
        parameter=q_initial,
        model_args={'h': h, 'u0': u_initial, 'tol': tol},
        dirichlet_ids=dirichlet_ids
    )

    solver = solver_type(problem, callback)

    max_iterations = 100
    iterations = solver.solve(rtol=1e-2, atol=0, max_iterations=max_iterations)
    print('Number of iterations: {}'.format(iterations))

    assert iterations < max_iterations
    q = solver.parameter
    assert firedrake.norm(q - q_true)/firedrake.norm(q_initial - q_true) < 1/3

