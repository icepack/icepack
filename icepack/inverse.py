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

import numpy as np
import scipy.optimize
import firedrake
from firedrake import action, adjoint, replace, ln, dx


def _bracket(f):
    """Given a real function of a single variable, return a value `t` such that
    `f(t) < f(0)`, which can then be used for a more thorough line search"""
    f_0 = f(0)
    t = 1.0
    while True:
        try:
            f_t = f(t)
            if f_t < f_0:
                return t
        except (AssertionError, firedrake.ConvergenceError):
            pass

        t /= 2


class InverseProblem(object):
    """Stores data for estimating a parameter of a model from observed data

    This object stores all of the data needed to estimate a parameter, such
    as the rheology or friction coefficient of a glacier, from remote
    sensing measurements. The parameter is estimated through an iterative
    minimization process; the key variables that need to be stored are the
    current guess for the parameter, the velocity field computed from that
    parameter, and the adjoint state field.

    At present, this class assumes that the model physics are an elliptic
    PDE that arises from an action principle. It is not equipped to deal
    with, say, the mass conservation equation, which is hyperbolic.
    """

    def __init__(self, model, method, objective, regularization,
                 state_name, state, parameter_name, parameter,
                 model_args={}, dirichlet_ids=[],
                 callback=None):
        """Initialize the inverse problem

        Parameters
        ----------
        model
            The forward model physics
        method
            The method of `model` to solve the forward model physics
        objective : firedrake.Form
            A functional of the state variable; this is what we will
            minimize
        regularization : firedrake.Form
            A functional of the parameter variable that penalizes
            unphysical behavior
        state_name : str
            The name of the state variable as expected by `model.solve`
        state : firedrake.Function
            The initial value of the state variable
        parameter_name : str
            The name of the parameter variable as expected by `model.solve`
        parameter : firedrake.Function
            The initial value of the parameter variable
        model_args : dict, optional
            Any additional arguments to `model.solve`
        dirichlet_ids : list of int, optional
            IDs of points on the domain boundary where Dirichlet conditions
            are applied

        The state variable must be an argument of the objective functional,
        and the parameter variable must be an argument of the
        regularization functional.
        """
        self.model = model
        self.method = method

        self.model_args = model_args
        self.dirichlet_ids = dirichlet_ids

        self.parameter_name = parameter_name
        self.parameter = parameter
        self.state_name = state_name
        self.state = state

        self.objective = objective
        self.regularization = regularization


class InverseSolver(object):
    """Base class for approximating the solution of an inverse problem"""
    def _setup(self, problem, callback=(lambda s: None)):
        self._problem = problem
        self._callback = callback

        self._p = problem.parameter.copy(deepcopy=True)
        self._u = problem.state.copy(deepcopy=True)

        self._model_args = dict(**problem.model_args,
                                dirichlet_ids=problem.dirichlet_ids)
        u_name, p_name = problem.state_name, problem.parameter_name
        args = dict(**self._model_args, **{u_name: self._u, p_name: self._p})

        # Make the form compiler use a reasonable number of quadrature points
        degree = problem.model.quadrature_degree(**args)
        self._fc_params = {'quadrature_degree': degree}

        # Create the error, regularization, and barrier functionals
        self._E = replace(problem.objective, {problem.state: self._u})
        self._R = replace(problem.regularization, {problem.parameter: self._p})
        self._J = self._E + self._R

        # Create the weak form of the forward model, the adjoint state, and
        # the derivative of the objective functional
        self._F = firedrake.derivative(problem.model.action(**args), self._u)
        self._dF_du = firedrake.derivative(self._F, self._u)

        # Create a search direction
        dR = firedrake.derivative(self._R, self._p)
        self._solver_params = {'ksp_type': 'preonly', 'pc_type': 'lu'}
        Q = self._p.function_space()
        self._q = firedrake.Function(Q)

        # Create the adjoint state variable
        V = self.state.function_space()
        self._λ = firedrake.Function(V)
        dF_dp = firedrake.derivative(self._F, self._p)

        # Create Dirichlet BCs where they apply for the adjoint solve
        rank = self._λ.ufl_element().num_sub_elements()
        zero = 0 if rank == 0 else (0,) * rank
        self._bc = firedrake.DirichletBC(V, zero, problem.dirichlet_ids)

        # Create the derivative of the objective functional
        self._dE = firedrake.derivative(self._E, self._u)
        dR = firedrake.derivative(self._R, self._p)
        self._dJ = (action(adjoint(dF_dp), self._λ) + dR)

    @property
    def problem(self):
        """The instance of the inverse problem we're solving"""
        return self._problem

    @property
    def parameter(self):
        """The current value of the parameter we're estimating"""
        return self._p

    @property
    def state(self):
        """The state variable computed from the current value of the
        parameter"""
        return self._u

    @property
    def adjoint_state(self):
        """The adjoint state variable computed from the current value of
        the parameters and the primal state"""
        return self._λ

    @property
    def search_direction(self):
        """Return the direction along which we'll search for a new value of
        the parameters"""
        return self._q

    @property
    def objective(self):
        """The functional of the state variable that we're minimizing"""
        return self._E

    @property
    def regularization(self):
        """The regularization functional, which penalizes unphysical modes
        in the inferred parameter"""
        return self._R

    @property
    def gradient(self):
        """The derivative of the Lagrangian (objective + regularization +
        physics constraints) with respect to the parameter"""
        return self._dJ

    def _forward_solve(self, p):
        method = self.problem.method
        model = self.problem.model
        args = self._model_args
        return method(model, **args, **{self.problem.parameter_name: p})

    def _assemble(self, *args, **kwargs):
        return firedrake.assemble(*args, **kwargs,
                                  form_compiler_parameters=self._fc_params)

    def update_state(self):
        """Update the observable state for a new value of the parameters"""
        u, p = self.state, self.parameter
        u.assign(self._forward_solve(p))

    def update_adjoint_state(self):
        """Update the adjoint state for new values of the observable state and
        parameters so that we can calculate derivatives"""
        λ = self.adjoint_state
        L = firedrake.adjoint(self._dF_du)
        firedrake.solve(L == -self._dE, λ, self._bc,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

    def line_search(self):
        """Perform a line search along the descent direction to get a new
        value of the parameter"""
        u, p, q = self.state, self.parameter, self.search_direction

        s = firedrake.Constant(0)
        p_s = p + s * q
        u_s = u.copy(deepcopy=True)

        def f(t):
            s.assign(t)
            u_s.assign(self._forward_solve(p_s))
            return self._assemble(replace(self._J, {u: u_s, p: p_s}))

        try:
            line_search_options = self._line_search_options
        except AttributeError:
            line_search_options = {}

        bracket = scipy.optimize.bracket(f, xa=0.0, xb=_bracket(f))[:3]
        result = scipy.optimize.minimize_scalar(f, bracket=bracket,
                                                options=line_search_options)

        if not result.success:
            raise ValueError("Line search failed: {}".format(result.message))

        return result.x

    def step(self):
        """Perform a line search along the current descent direction to get
        a new value of the parameters, then compute the new state, adjoint
        and descent direction."""
        p, q = self.parameter, self.search_direction
        t = self.line_search()
        p += t * q
        self.update_state()
        self.update_adjoint_state()
        self.update_search_direction()
        self._callback(self)

    def solve(self, atol=0.0, rtol=1e-6, max_iterations=None):
        """Search for a new value of the parameters, stopping once either
        the objective functional gets below a threshold value or stops
        improving."""
        max_iterations = max_iterations or np.inf
        J_initial = np.inf

        for iteration in range(max_iterations):
            J = self._assemble(self._J)
            if ((J_initial - J) < rtol * J_initial) or (J <= atol):
                return iteration
            J_initial = J

            self.step()

        return max_iterations


class GradientDescentSolver(InverseSolver):
    """Uses line search along the objective function gradient"""
    def __init__(self, problem, callback=(lambda s: None)):
        """Initializes the inverse solver with the right functionals and
        auxiliary fields

        Parameters
        ----------
        problem : InverseProblem
            The instance of the problem to be solved
        callback : callable, optional
            Function to call at the end of every iteration
        """
        self._setup(problem, callback)

        # Get the solver object into a consistent internal state
        self.update_state()
        self.update_adjoint_state()
        self.update_search_direction()

        # Call the post-iteration function for the first time
        self._callback(self)

    def update_search_direction(self):
        """Set the search direction to be the inverse of the mass matrix times
        the gradient of the objective"""
        q, dJ = self.search_direction, self.gradient
        Q = q.function_space()
        M = firedrake.TrialFunction(Q) * firedrake.TestFunction(Q) * dx
        firedrake.solve(M == -dJ, q,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)


class GaussNewtonSolver(InverseSolver):
    def __init__(self, problem, callback=(lambda s: None),
                 search_tolerance=1e-3):
        self._setup(problem, callback)
        self.update_state()
        self.update_adjoint_state()

        self._search_tolerance = search_tolerance
        self._line_search_options = {'xtol': search_tolerance / 2}
        self.update_search_direction()

        self._callback(self)

    def gauss_newton_mult(self, q):
        u, p = self.state, self.parameter

        dE = firedrake.derivative(self._E, u)
        d2E = firedrake.derivative(dE, u)
        dR = firedrake.derivative(self._R, p)
        dF_du, dF_dp = self._dF_du, firedrake.derivative(self._F, p)

        w = firedrake.Function(u.function_space())
        firedrake.solve(dF_du == firedrake.action(dF_dp, q), w, self._bc,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

        v = firedrake.Function(u.function_space())
        firedrake.solve(firedrake.adjoint(dF_du) == firedrake.action(d2E, w),
                        v, self._bc,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

        return firedrake.action(firedrake.adjoint(dF_dp), v) + \
            firedrake.derivative(dR, p, q)

    def gauss_newton_energy_norm(self, q):
        u, p = self.state, self.parameter

        dE = firedrake.derivative(self._E, u)
        d2E = firedrake.derivative(dE, u)
        dR = firedrake.derivative(self._R, p)
        d2R = firedrake.derivative(dR, p)
        dF_du, dF_dp = self._dF_du, firedrake.derivative(self._F, p)

        v = firedrake.Function(u.function_space())
        firedrake.solve(dF_du == firedrake.action(dF_dp, q), v, self._bc,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

        return self._assemble(firedrake.energy_norm(d2E, v) +
                              firedrake.energy_norm(d2R, q))

    def update_search_direction(self):
        p, q, dJ = self.parameter, self.search_direction, self.gradient

        dR = firedrake.derivative(self.regularization, self.parameter)
        Q = q.function_space()
        M = firedrake.TrialFunction(Q) * firedrake.TestFunction(Q) * dx + \
            firedrake.derivative(dR, p)

        # Compute the preconditioned residual
        z = firedrake.Function(Q)
        firedrake.solve(M == -dJ, z,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

        # This variable is a search direction for a search direction, which
        # is definitely not confusing at all.
        s = z.copy(deepcopy=True)
        q *= 0.0

        old_cost = np.inf
        while True:
            z_mnorm = self._assemble(firedrake.energy_norm(M, z))
            s_hnorm = self.gauss_newton_energy_norm(s)
            α = z_mnorm / s_hnorm

            δz = firedrake.Function(Q)
            g = self.gauss_newton_mult(s)
            firedrake.solve(M == g, δz,
                            solver_parameters=self._solver_params,
                            form_compiler_parameters=self._fc_params)

            q += α * s
            z -= α * δz

            β = self._assemble(firedrake.energy_norm(M, z)) / z_mnorm
            s *= β
            s += z

            energy_norm = self.gauss_newton_energy_norm(q)
            cost = 0.5 * energy_norm + self._assemble(firedrake.action(dJ, q))

            if (abs(old_cost - cost) / (0.5 * energy_norm)
                    < self._search_tolerance):
                return

            old_cost = cost
