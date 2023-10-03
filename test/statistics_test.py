# Copyright (C) 2022 by Daniel Shapero <shapero@uw.edu>
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
import firedrake.adjoint
import icepack
from icepack.statistics import StatisticsProblem, MaximumProbabilityEstimator
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    glen_flow_law as n,
)


@pytest.mark.skipif(not icepack.statistics.has_rol, reason="Couldn't import ROL")
def test_poisson_problem():
    Nx, Ny = 32, 32

    mesh = firedrake.UnitSquareMesh(Nx, Ny)
    degree = 2
    Q = firedrake.FunctionSpace(mesh, "CG", degree)
    V = firedrake.FunctionSpace(mesh, "CG", degree)

    x, y = firedrake.SpatialCoordinate(mesh)
    f = interpolate(firedrake.Constant(1), Q)

    def simulation(q):
        # The momentum balance equations in icepack are formulated as a
        # minimization problem, so for consistency we use the minimization
        # form of the Poisson problem here.
        u = firedrake.Function(V)
        J = (0.5 * exp(q) * inner(grad(u), grad(u)) - f * u) * dx
        F = firedrake.derivative(J, u)

        bc = firedrake.DirichletBC(V, 0, "on_boundary")
        problem = firedrake.NonlinearVariationalProblem(F, u, bc)
        params = {
            "solver_parameters": {
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        }
        solver = firedrake.NonlinearVariationalSolver(problem, **params)
        solver.solve()

        return u

    q_true = interpolate(-4 * ((x - 0.5) ** 2 + (y - 0.5) ** 2), Q)
    u_obs = simulation(q_true)

    def loss_functional(u):
        return 0.5 * (u - u_obs) ** 2 * dx

    α = firedrake.Constant(1e-4)

    def regularization(q):
        return 0.5 * α**2 * inner(grad(q), grad(q)) * dx

    q_initial = firedrake.Function(Q)
    problem = StatisticsProblem(simulation, loss_functional, regularization, q_initial)
    estimator = MaximumProbabilityEstimator(problem, gradient_tolerance=1e-7)
    q = estimator.solve()

    assert firedrake.norm(q - q_true) < 0.25


@pytest.mark.skipif(not icepack.statistics.has_rol, reason="Couldn't import ROL")
@pytest.mark.parametrize("with_noise", [False, True])
def test_ice_shelf_inverse(with_noise):
    Nx, Ny = 32, 32
    Lx, Ly = 20e3, 20e3

    u0 = 100
    h0, δh = 500, 100
    T0, δT = 254.15, 5.0
    A0 = icepack.rate_factor(T0)
    δA = icepack.rate_factor(T0 + δT) - A0

    def exact_u(x):
        ρ = ρ_I * (1 - ρ_I / ρ_W)
        Z = icepack.rate_factor(T0) * (ρ * g * h0 / 4) ** n
        q = 1 - (1 - (δh / h0) * (x / Lx)) ** (n + 1)
        δu = Z * q * Lx * (h0 / δh) / (n + 1)
        return u0 + δu

    mesh = firedrake.RectangleMesh(Nx, Ny, Lx, Ly)
    degree = 2
    V = firedrake.VectorFunctionSpace(mesh, "CG", degree)
    Q = firedrake.FunctionSpace(mesh, "CG", degree)

    x, y = firedrake.SpatialCoordinate(mesh)
    u_initial = interpolate(as_vector((exact_u(x), 0)), V)
    q_initial = interpolate(firedrake.Constant(0), Q)
    h = interpolate(h0 - δh * x / Lx, Q)

    def viscosity(**kwargs):
        u = kwargs["velocity"]
        h = kwargs["thickness"]
        q = kwargs["log_fluidity"]

        A = A0 * firedrake.exp(-q / n)
        return icepack.models.viscosity.viscosity_depth_averaged(
            velocity=u, thickness=h, fluidity=A
        )

    model = icepack.models.IceShelf(viscosity=viscosity)
    dirichlet_ids = [1, 3, 4]
    flow_solver = icepack.solvers.FlowSolver(
        model,
        dirichlet_ids=dirichlet_ids,
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters={
            "snes_type": "newtonls",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    r = firedrake.sqrt((x / Lx - 1 / 2) ** 2 + (y / Ly - 1 / 2) ** 2)
    R = 1 / 4
    expr = firedrake.max_value(0, δA * (1 - (r / R) ** 2))
    q_true = firedrake.interpolate(-n * firedrake.ln(1 + expr / A0), Q)
    u_true = flow_solver.diagnostic_solve(
        velocity=u_initial, thickness=h, log_fluidity=q_true
    )

    u_obs = u_true.copy(deepcopy=True)
    area = firedrake.assemble(firedrake.Constant(1) * dx(mesh))
    σ = firedrake.Constant(1.0)
    L = firedrake.Constant(1e-4 * Lx)
    if with_noise:
        σ = firedrake.Constant(0.001 * firedrake.norm(u_true) / np.sqrt(area))
        shape = u_obs.dat.data_ro.shape
        u_obs.dat.data[:] += float(σ) * random.standard_normal(shape) / np.sqrt(2)
        L = firedrake.Constant(0.25 * Lx)

    def loss_functional(u):
        return 0.5 * ((u - u_obs) / σ) ** 2 * dx

    def regularization(q):
        return 0.5 * L**2 * inner(grad(q), grad(q)) * dx

    def simulation(q):
        return flow_solver.diagnostic_solve(
            velocity=u_initial, thickness=h, log_fluidity=q
        )

    stats_problem = StatisticsProblem(
        simulation, loss_functional, regularization, q_initial
    )
    estimator = MaximumProbabilityEstimator(stats_problem)
    q = estimator.solve()

    assert firedrake.norm(q - q_true) / firedrake.norm(q_initial - q_true) < 0.25
