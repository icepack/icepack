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

import firedrake
from .utilities import default_solver_parameters


class MinimizationProblem:
    def __init__(self, E, S, u, bcs, form_compiler_parameters):
        r"""Nonlinear optimization problem argmin E(u).

        Parameters
        ----------
        E : firedrake.Form
            The functional to be minimized
        S : firedrake.Form
            A positive functional measure the scale of the solution
        u : firedrake.Function
            Initial guess for the minimizer
        bcs : firedrake.DirichletBC
            any essential boundary conditions for the solution
        """
        self.E = E
        self.S = S
        self.u = u
        self.bcs = bcs
        self.form_compiler_parameters = form_compiler_parameters

    def assemble(self, *args, **kwargs):
        kwargs["form_compiler_parameters"] = self.form_compiler_parameters
        return firedrake.assemble(*args, **kwargs)


class NewtonSolver:
    def __init__(self, problem, tolerance, solver_parameters=None, **kwargs):
        r"""Solve a MinimizationProblem using Newton's method with backtracking
        line search

        Parameters
        ----------
        problem : MinimizationProblem
            The particular problem instance to solve
        tolerance : float
            dimensionless tolerance for when to stop iterating, measured with
            with respect to the problem's scale functional
        solver_parameters : dict (optional)
            Linear solve parameters for computing the search direction
        armijo : float (optional)
            Parameter in the Armijo condition for line search; defaults to
            1e-4, see Nocedal and Wright
        contraction : float (optional)
            shrinking factor for backtracking line search; defaults to .5
        max_iterations : int (optional)
            maximum number of outer-level Newton iterations; defaults to 50
        """
        self.problem = problem
        self.tolerance = tolerance
        if solver_parameters is None:
            solver_parameters = default_solver_parameters

        self.armijo = kwargs.pop("armijo", 1e-4)
        self.contraction = kwargs.pop("contraction", 0.5)
        self.max_iterations = kwargs.pop("max_iterations", 50)

        u = self.problem.u
        V = u.function_space()
        v = firedrake.Function(V)
        self.v = v

        E = self.problem.E
        self.F = firedrake.derivative(E, u)
        self.J = firedrake.derivative(self.F, u)
        self.dE_dv = firedrake.action(self.F, v)

        bcs = None
        if self.problem.bcs:
            bcs = firedrake.homogenize(self.problem.bcs)
        problem = firedrake.LinearVariationalProblem(
            self.J,
            -self.F,
            v,
            bcs,
            constant_jacobian=False,
            form_compiler_parameters=self.problem.form_compiler_parameters,
        )
        self.search_direction_solver = firedrake.LinearVariationalSolver(
            problem, solver_parameters=solver_parameters
        )

        self.search_direction_solver.solve()
        self.t = firedrake.Constant(0.0)
        self.iteration = 0

    def reinit(self):
        self.search_direction_solver.solve()
        self.t.assign(0.0)
        self.iteration = 0

    def step(self):
        r"""Perform a backtracking line search for the next value of the
        solution and compute the search direction for the next step"""
        E = self.problem.E
        u = self.problem.u
        v = self.v
        t = self.t

        t.assign(1.0)
        E_0 = self.problem.assemble(E)
        slope = self.problem.assemble(self.dE_dv)
        if slope > 0:
            raise firedrake.ConvergenceError(
                "Minimization solver has invalid search direction. This is "
                "likely due to a negative thickness or friction coefficient or"
                "otherwise physically invalid input data."
            )

        E_t = firedrake.replace(E, {u: u + t * v})

        armijo = self.armijo
        contraction = self.contraction
        while self.problem.assemble(E_t) > E_0 + armijo * float(t) * slope:
            t.assign(t * contraction)

        u.assign(u + t * v)
        self.search_direction_solver.solve()
        self.iteration += 1

    def solve(self):
        r"""Step the Newton iteration until convergence"""
        self.reinit()

        dE_dv = self.dE_dv
        S = self.problem.S
        _assemble = self.problem.assemble
        while abs(_assemble(dE_dv)) > self.tolerance * _assemble(S):
            self.step()
            if self.iteration >= self.max_iterations:
                raise firedrake.ConvergenceError(
                    f"Newton search did not converge after {self.max_iterations} iterations!"
                )
