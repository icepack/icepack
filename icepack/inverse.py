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

r"""Solvers for inverse problems

This module contains objects for specifying and solving inverse problems, where
some unobservable field is estimated based on observations of an observable
field and a physical model that connects the two. The class `InverseProblem` is
used to specify the problem to be solved, while the classes that inherit from
`InverseSolver` are used to solve it.
"""

import numpy as np
from scipy.optimize import bracket, minimize_scalar
import firedrake
from firedrake import action, adjoint, derivative, replace, dx, Constant
from .solvers import FlowSolver
from .utilities import default_solver_parameters


def _bracket(f, max_iterations):
    r"""Given a decreasing real function of a single variable, return a value
    `t` such that `f(t) < f(0)`, which can then be used for a more thorough
    line search"""
    f_0 = f(0)
    t = 1.0
    for iteration in range(max_iterations):
        try:
            f_t = f(t)
            if f_t < f_0:
                return t
        except (AssertionError, firedrake.ConvergenceError):
            pass

        t /= 2

    return t


class InverseProblem:
    r"""Specifies an inverse problem

    This object is used to specify an inverse problem, i.e. estimating a
    parameter :math:`p` from measurements of a field :math:`u`, where the two
    are linked by some model physics. It consists mainly of a description of
    the physics model, the model-data misfit and regularization functionals,
    the initial state and parameters, and any auxiliary data such as extra
    fields or boundary conditions.

    This object merely specifies what problem to solve, not how to solve it.
    That functionality is contained in the class `InverseSolver` and the
    classes that inherit from it, such as `GradientDescentSolver` and
    `GaussNewtonSolver`.

    At present, this class assumes that the model physics are an elliptic
    PDE that arises from an action principle. It is not equipped to deal
    with, say, the mass conservation equation, which is hyperbolic.
    """

    def __init__(
        self,
        model,
        objective,
        regularization,
        state_name,
        state,
        parameter_name,
        parameter,
        solver_type=FlowSolver,
        solver_kwargs={},
        diagnostic_solve_kwargs={},
    ):
        r"""Initialize the inverse problem

        Parameters
        ----------
        model
            The forward model physics; either ``IceShelf``, ``IceStream``, etc.
        objective
            A python function that takes in the velocity field and returns
            the model-data misfit functional
        regularization
            A python function that takes in the field to infer and returns
            the regularization functional, i.e. the penalty for unphysical
            parameter values
        state_name : str
            The name of the state variable as expected by `model.solve`
        state : firedrake.Function
            The initial value of the state variable
        parameter_name : str
            The name of the parameter variable as expected by `model.solve`
        parameter : firedrake.Function
            The initial value of the parameter variable
        solver_type: class, optional
            The type of the solver object for the diagnostic equations; must
            have a ``diagnostic_solve`` method
        solver_kwargs : dict, optional
            Any additional arguments to initialize the solver, i.e. tolerances
            or boundary IDs
        diagnostic_solve_kwargs : dict, optional
            Any additional arguments to pass when calling the diagnostic solve
            procedure, including additional physical fields

        The state variable must be an argument of the objective functional,
        and the parameter variable must be an argument of the
        regularization functional.
        """
        self.model = model
        self.solver_type = solver_type
        self.solver_kwargs = solver_kwargs
        self.diagnostic_solve_kwargs = diagnostic_solve_kwargs
        self.dirichlet_ids = solver_kwargs.get("dirichlet_ids", [])

        self.parameter_name = parameter_name
        self.parameter = parameter
        self.state_name = state_name
        self.state = state

        self.objective = objective
        self.regularization = regularization


class InverseSolver:
    r"""Base class for approximating the solution of an inverse problem

    This object stores most of the data needed to iteratively optimize the
    value of some parameter of a model, such as the rheology or friction of a
    glacier, to match remote sensing observations. The key variables that need
    to be stored are the current guess for the parameter, the observable state
    computed from that parameter, the adjoint state, and the search direction.

    The optimization problem is solved using a line search method; at each step
    the parameter :math:`p_k` is updated by finding a value :math:`\alpha_k`
    such that

    .. math::
        p_{k + 1} = p_k + \alpha_k\cdot q_k

    reduces the value of the objective function, where :math:`q_k` is the
    search direction. This object implements most of the procedures necessary
    for keeping the parameter, state, and adjoint consistent. Objects that
    inherit from this one only need to define how the search direction is
    computed.
    """

    def _setup(self, problem, callback=(lambda s: None)):
        self._problem = problem
        self._callback = callback

        self._p = problem.parameter.copy(deepcopy=True)
        self._u = problem.state.copy(deepcopy=True)

        self._solver = self.problem.solver_type(
            self.problem.model, **self.problem.solver_kwargs
        )
        u_name, p_name = problem.state_name, problem.parameter_name
        solve_kwargs = dict(
            **problem.diagnostic_solve_kwargs, **{u_name: self._u, p_name: self._p}
        )

        # Make the form compiler use a reasonable number of quadrature points
        degree = problem.model.quadrature_degree(**solve_kwargs)
        self._fc_params = {"quadrature_degree": degree}

        # Create the error, regularization, and barrier functionals
        self._E = problem.objective(self._u)
        self._R = problem.regularization(self._p)
        self._J = self._E + self._R

        # Create the weak form of the forward model, the adjoint state, and
        # the derivative of the objective functional
        A = problem.model.action(**solve_kwargs)
        self._F = derivative(A, self._u)
        self._dF_du = derivative(self._F, self._u)

        # Create a search direction
        dR = derivative(self._R, self._p)
        # TODO: Make this customizable
        self._solver_params = default_solver_parameters
        Q = self._p.function_space()
        self._q = firedrake.Function(Q)

        # Create the adjoint state variable
        V = self.state.function_space()
        self._λ = firedrake.Function(V)
        dF_dp = derivative(self._F, self._p)

        # Create Dirichlet BCs where they apply for the adjoint solve
        rank = self._λ.ufl_element().num_sub_elements()
        if rank == 0:
            zero = Constant(0)
        else:
            zero = firedrake.as_vector((0,) * rank)
        self._bc = firedrake.DirichletBC(V, zero, problem.dirichlet_ids)

        # Create the derivative of the objective functional
        self._dE = derivative(self._E, self._u)
        dR = derivative(self._R, self._p)
        self._dJ = action(adjoint(dF_dp), self._λ) + dR

        # Create problem and solver objects for the adjoint state
        L = adjoint(self._dF_du)
        adjoint_problem = firedrake.LinearVariationalProblem(
            L,
            -self._dE,
            self._λ,
            self._bc,
            form_compiler_parameters=self._fc_params,
            constant_jacobian=False,
        )
        self._adjoint_solver = firedrake.LinearVariationalSolver(
            adjoint_problem, solver_parameters=self._solver_params
        )

    @property
    def problem(self):
        r"""The instance of the inverse problem we're solving"""
        return self._problem

    @property
    def parameter(self):
        r"""The current value of the parameter we're estimating"""
        return self._p

    @property
    def state(self):
        r"""The state variable computed from the current value of the
        parameter"""
        return self._u

    @property
    def adjoint_state(self):
        r"""The adjoint state variable computed from the current value of
        the parameters and the primal state"""
        return self._λ

    @property
    def search_direction(self):
        r"""Return the direction along which we'll search for a new value of
        the parameters"""
        return self._q

    @property
    def objective(self):
        r"""The functional of the state variable that we're minimizing"""
        return self._E

    @property
    def regularization(self):
        r"""The regularization functional, which penalizes unphysical modes
        in the inferred parameter"""
        return self._R

    @property
    def gradient(self):
        r"""The derivative of the Lagrangian (objective + regularization +
        physics constraints) with respect to the parameter"""
        return self._dJ

    def _forward_solve(self, p):
        solver = self._solver
        problem = self.problem
        state_name, parameter_name = problem.state_name, problem.parameter_name
        kwargs = dict(
            **{state_name: self.state, parameter_name: p},
            **self.problem.diagnostic_solve_kwargs,
        )
        return solver.diagnostic_solve(**kwargs)

    def _assemble(self, *args, **kwargs):
        return firedrake.assemble(
            *args, **kwargs, form_compiler_parameters=self._fc_params
        )

    def update_state(self):
        r"""Update the observable state for a new value of the parameters"""
        u, p = self.state, self.parameter
        u.assign(self._forward_solve(p))

    def update_adjoint_state(self):
        r"""Update the adjoint state for new values of the observable state and
        parameters so that we can calculate derivatives"""
        self._adjoint_solver.solve()

    def line_search(self):
        r"""Perform a line search along the descent direction to get a new
        value of the parameter"""
        u, p, q = self.state, self.parameter, self.search_direction
        u_t, p_t = u.copy(deepcopy=True), p.copy(deepcopy=True)

        def f(t):
            p_t.assign(p + firedrake.Constant(t) * q)
            u_t.assign(self._forward_solve(p_t))
            return self._assemble(replace(self._J, {u: u_t, p: p_t}))

        try:
            line_search_options = self._line_search_options
        except AttributeError:
            line_search_options = {}

        brack = bracket(f, xa=0.0, xb=_bracket(f, max_iterations=30))[:3]
        result = minimize_scalar(f, bracket=brack, options=line_search_options)

        if not result.success:
            raise ValueError(f"Line search failed: {result.message}")

        return result.x

    def step(self):
        r"""Perform a line search along the current descent direction to get
        a new value of the parameters, then compute the new state, adjoint,
        and descent direction."""
        p, q = self.parameter, self.search_direction
        t = self.line_search()
        p.assign(p + Constant(t) * q)
        self.update_state()
        self.update_adjoint_state()
        self.update_search_direction()
        self._callback(self)

    def solve(self, atol=0.0, rtol=1e-6, etol=0.0, max_iterations=200):
        r"""Search for a new value of the parameters, stopping once either
        the objective functional gets below a threshold value or stops
        improving.

        Parameters
        ----------
        atol : float
            Absolute stopping tolerance; stop iterating when the objective
            drops below this value
        rtol : float
            Relative stopping tolerance; stop iterating when the relative
            decrease in the objective drops below this value
        etol : float
            Expectation stopping tolerance; stop iterating when the relative
            expected decrease in the objective from the Newton decrement drops
            below this value
        max_iterations : int
            Maximum number of iterations to take
        """
        J_initial = np.inf

        for iteration in range(max_iterations):
            J = self._assemble(self._J)

            q = self.search_direction
            dJ_dq = self._assemble(firedrake.action(self.gradient, q))

            if (
                ((J_initial - J) < rtol * J_initial)
                or (-dJ_dq < etol * J)
                or (J <= atol)
            ):
                return iteration

            J_initial = J
            self.step()

        return max_iterations


class GradientDescentSolver(InverseSolver):
    r"""Implementation of `InverseSolver` using the objective function gradient
    directly for a search direction

    This implementation of inverse solvers uses the search direction

    .. math::
        q = -M^{-1}dJ

    where :math:`M` is the finite element mass matrix and :math:`dJ` is the
    gradient of the objective functional. The search direction is easy to
    compute using this method, but is often poorly scaled, resulting in more
    expensive bracketing and line search phases."""

    def __init__(self, problem, callback=(lambda s: None)):
        self._setup(problem, callback)
        self.update_state()
        self.update_adjoint_state()

        q, dJ = self.search_direction, self.gradient
        Q = q.function_space()
        M = firedrake.TrialFunction(Q) * firedrake.TestFunction(Q) * dx
        problem = firedrake.LinearVariationalProblem(
            M, -dJ, q, form_compiler_parameters=self._fc_params
        )
        self._search_direction_solver = firedrake.LinearVariationalSolver(
            problem, solver_parameters=self._solver_params
        )

        self.update_search_direction()
        self._callback(self)

    def update_search_direction(self):
        r"""Set the search direction to be the inverse of the mass matrix times
        the gradient of the objective"""
        self._search_direction_solver.solve()


class GaussNewtonCG:
    def __init__(self, solver):
        r"""State machine for solving the Gauss-Newton subproblem via the
        preconditioned conjugate gradient method"""
        self._assemble = solver._assemble
        u = solver.state
        p = solver.parameter
        E = solver._E
        dE = derivative(E, u)
        R = solver._R
        dR = derivative(R, p)
        F = solver._F
        dF_du = derivative(F, u)
        dF_dp = derivative(F, p)
        # TODO: Make this an arbitrary RHS -- the solver can set it to the
        # gradient if we want
        dJ = solver.gradient
        bc = solver._bc

        V = u.function_space()
        Q = p.function_space()

        # Create the preconditioned residual and solver
        z = firedrake.Function(Q)
        s = firedrake.Function(Q)
        φ, ψ = firedrake.TestFunction(Q), firedrake.TrialFunction(Q)
        M = φ * ψ * dx + derivative(dR, p)
        residual_problem = firedrake.LinearVariationalProblem(
            M,
            -dJ,
            z,
            form_compiler_parameters=solver._fc_params,
            constant_jacobian=False,
        )
        residual_solver = firedrake.LinearVariationalSolver(
            residual_problem, solver_parameters=solver._solver_params
        )

        self._preconditioner = M
        self._residual = z
        self._search_direction = s
        self._residual_solver = residual_solver

        # Create a variable to store the current solution of the Gauss-Newton
        # problem and the solutions of the auxiliary tangent sub-problems
        q = firedrake.Function(Q)
        v = firedrake.Function(V)
        w = firedrake.Function(V)

        # Create linear problem and solver objects for the auxiliary tangent
        # sub-problems
        tangent_linear_problem = firedrake.LinearVariationalProblem(
            dF_du,
            action(dF_dp, s),
            w,
            bc,
            form_compiler_parameters=solver._fc_params,
            constant_jacobian=False,
        )
        tangent_linear_solver = firedrake.LinearVariationalSolver(
            tangent_linear_problem, solver_parameters=solver._solver_params
        )

        adjoint_tangent_linear_problem = firedrake.LinearVariationalProblem(
            adjoint(dF_du),
            derivative(dE, u, w),
            v,
            bc,
            form_compiler_parameters=solver._fc_params,
            constant_jacobian=False,
        )
        adjoint_tangent_linear_solver = firedrake.LinearVariationalSolver(
            adjoint_tangent_linear_problem, solver_parameters=solver._solver_params
        )

        self._rhs = dJ
        self._solution = q
        self._tangent_linear_solution = w
        self._tangent_linear_solver = tangent_linear_solver
        self._adjoint_tangent_linear_solution = v
        self._adjoint_tangent_linear_solver = adjoint_tangent_linear_solver

        self._product = action(adjoint(dF_dp), v) + derivative(dR, p, s)

        # Create the update to the residual and the associated solver
        δz = firedrake.Function(Q)
        Gs = self._product
        delta_residual_problem = firedrake.LinearVariationalProblem(
            M,
            Gs,
            δz,
            form_compiler_parameters=solver._fc_params,
            constant_jacobian=False,
        )
        delta_residual_solver = firedrake.LinearVariationalSolver(
            delta_residual_problem, solver_parameters=solver._solver_params
        )

        self._delta_residual = δz
        self._delta_residual_solver = delta_residual_solver

        self._residual_energy = 0.0
        self._search_direction_energy = 0.0

        self.reinit()

    def reinit(self):
        r"""Restart the solution to 0"""
        self._iteration = 0

        M = self._preconditioner
        z = self.residual
        s = self.search_direction
        Gs = self.operator_product
        Q = z.function_space()

        self.solution.assign(firedrake.Function(Q))
        self._residual_solver.solve()
        s.assign(z)
        self._residual_energy = self._assemble(firedrake.energy_norm(M, z))

        self.update_state()
        self._search_direction_energy = self._assemble(action(Gs, s))

        self._energy = 0.0
        self._objective = 0.0

    @property
    def iteration(self):
        r"""The number of iterations executed"""
        return self._iteration

    @property
    def solution(self):
        r"""The current guess for the solution"""
        return self._solution

    @property
    def tangent_linear_solution(self):
        r"""The solution of the tangent linear system"""
        return self._tangent_linear_solution

    @property
    def adjoint_tangent_linear_solution(self):
        r"""The solution of the adjoint of the tangent linear system"""
        return self._adjoint_tangent_linear_solution

    @property
    def operator_product(self):
        r"""A form representing the product of the Gauss-Newton operator
        and the current solution guess"""
        return self._product

    @property
    def preconditioner(self):
        return self._preconditioner

    @property
    def residual(self):
        r"""The preconditioned residual for the current solution"""
        return self._residual

    @property
    def search_direction(self):
        r"""The search direction for finding the next solution iterate"""
        return self._search_direction

    @property
    def residual_energy(self):
        r"""The energy norm of the residual w.r.t. the preconditioner"""
        return self._residual_energy

    @property
    def search_direction_energy(self):
        r"""The energy norm of the search direction w.r.t. to the Gauss-
        Newton operator"""
        return self._search_direction_energy

    def update_state(self):
        r"""Update the auxiliary state variables and the residual change"""
        self._tangent_linear_solver.solve()
        self._adjoint_tangent_linear_solver.solve()
        self._delta_residual_solver.solve()

    def step(self):
        r"""Take one step of the conjugate gradient iteration"""
        q = self.solution
        s = self.search_direction
        z = self.residual
        δz = self._delta_residual
        α = self.residual_energy / self.search_direction_energy

        Gs = self.operator_product
        dJ = self._rhs
        delta_energy = α * (
            self._assemble(action(Gs, q)) + 0.5 * α * self.search_direction_energy
        )
        self._energy += delta_energy
        self._objective += delta_energy + α * self._assemble(action(dJ, s))

        q.assign(q + Constant(α) * s)
        z.assign(z - Constant(α) * δz)

        M = self.preconditioner
        residual_energy = self._assemble(firedrake.energy_norm(M, z))
        β = residual_energy / self.residual_energy
        s.assign(Constant(β) * s + z)

        self.update_state()
        self._residual_energy = residual_energy
        Gs = self.operator_product
        self._search_direction_energy = self._assemble(action(Gs, s))

        self._iteration += 1

    def solve(self, tolerance, max_iterations):
        r"""Run the iteration until the objective functional does not decrease
        to within tolerance"""
        objective = np.inf
        for iteration in range(max_iterations):
            if objective - self._objective <= tolerance * self._energy:
                return

            objective = self._objective
            self.step()

        raise firedrake.ConvergenceError(
            f"Gauss-Newton CG failed to converge after {max_iterations} steps!"
        )


class GaussNewtonSolver(InverseSolver):
    r"""Implementation of `InverseSolver` using an approximation to the Hessian
    of the objective functional to approach Newton-like efficiency

    This implementation of inverse solvers uses the search direction

    .. math::
        q = -H^{-1}dJ

    where :math:`H` is the Gauss-Newton approximation to the Hessian of the
    objective functional. If :math:`E` is the model-data misfit, :math:`R` is
    the regularization, and :math:`G` is the linearization of the parameter-to-
    observation map, then the Gauss-Newton matrix is

    .. math::
        H = dG^*\cdot d^2E\cdot dG + d^2R.

    This matrix consists of only those terms in the Hessian of the full
    objective functional that are of "first order", i.e. any terms involving
    :math:`d^2G` are dropped. This search direction is more expensive to solve
    for than in, say, gradient descent. However, it is almost always properly
    scaled to the dimensions of the problem and converges in far fewer
    iterations.
    """

    def __init__(
        self,
        problem,
        callback=(lambda s: None),
        search_tolerance=1e-6,
        search_max_iterations=100,
    ):
        self._setup(problem, callback)
        self.update_state()
        self.update_adjoint_state()

        self._search_tolerance = search_tolerance
        self._search_max_iterations = search_max_iterations
        self._line_search_options = {"xtol": search_tolerance / 2}

        self._search_solver = GaussNewtonCG(self)
        self.update_search_direction()

        self._callback(self)

    def update_search_direction(self):
        r"""Solve the Gauss-Newton system for the new search direction using
        the preconditioned conjugate gradient method"""
        self._search_solver.reinit()
        self._search_solver.solve(self._search_tolerance, self._search_max_iterations)
        self.search_direction.assign(self._search_solver.solution)


class BFGSSolver(InverseSolver):
    r"""Implementation of `InverseSolver` using the limited-memory BFGS method
    to compute a search direction

    This implementation of inverse solvers uses a search direction based on the
    last `m` values of the parameter and objective gradient to construct a low-
    rank approximation to the inverse of the Hessian of the objective. The
    resulting iteration exhibits superlinear convergence, while the search
    direction is only marginally more expensive to compute than the steepest
    descent direction.

    See chapters 6-7 of Nocedal and Wright, Numerical Optimization, 2nd ed.
    """

    def __init__(self, problem, callback=(lambda s: None), memory=5):
        self._setup(problem, callback)
        self.update_state()
        self.update_adjoint_state()

        Q = self.parameter.function_space()
        self._memory = memory

        q, dJ = self.search_direction, self.gradient
        M = firedrake.TrialFunction(Q) * firedrake.TestFunction(Q) * dx
        f = firedrake.Function(Q)
        problem = firedrake.LinearVariationalProblem(
            M, dJ, f, form_compiler_parameters=self._fc_params
        )
        self._search_direction_solver = firedrake.LinearVariationalSolver(
            problem, solver_parameters=self._solver_params
        )
        self._search_direction_solver.solve()
        q.assign(-f)

        self._f = f
        self._rho = []
        self._ps = [self.parameter.copy(deepcopy=True)]
        self._fs = [q.copy(deepcopy=True)]
        self._fs[-1] *= -1

        self._callback(self)

    @property
    def memory(self):
        r"""Return the number of previous iterations used to construct the low-
        rank approximation to the Hessian"""
        return self._memory

    def update_search_direction(self):
        r"""Apply the low-rank approximation of the Hessian inverse

        This procedure implements the two-loop recursion algorithm to apply the
        low-rank approximation of the Hessian inverse to the derivative of the
        objective functional. See Nocedal and Wright, Numerical Optimization,
        2nd ed., algorithm 7.4."""
        p, q = self.parameter, self.search_direction
        self._search_direction_solver.solve()
        f = self._f

        # Append the latest values of the parameters and the objective gradient
        # and compute the curvature factor
        ps, fs, ρ = self._ps, self._fs, self._rho
        ρ.append(1 / self._assemble((p - ps[-1]) * (f - fs[-1]) * dx))
        ps.append(p.copy(deepcopy=True))
        fs.append(f.copy(deepcopy=True))

        # Forget any old values of the parameters and objective gradient
        ps = ps[-(self.memory + 1) :]
        fs = fs[-(self.memory + 1) :]
        ρ = ρ[-self.memory :]

        g = f.copy(deepcopy=True)
        m = len(ρ)
        α = np.zeros(m)
        for i in range(m - 1, -1, -1):
            α[i] = ρ[i] * self._assemble(f * (ps[i + 1] - ps[i]) * dx)
            g.assign(g - Constant(α[i]) * (fs[i + 1] - fs[i]))

        r = g.copy(deepcopy=True)
        dp, df = ps[-1] - ps[-2], fs[-1] - fs[-2]
        r *= self._assemble(dp * df * dx) / self._assemble(df * df * dx)

        for i in range(m):
            β = ρ[i] * self._assemble((fs[i + 1] - fs[i]) * r * dx)
            r.assign(r + Constant(α[i] - β) * (ps[i + 1] - ps[i]))

        q.assign(-r)
