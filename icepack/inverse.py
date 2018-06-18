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
from firedrake import replace, dx


def _find_zero_crossing(p, q):
    if np.min(p.dat.data_ro[:]) <= 0:
        return 0.0

    t = 1.0
    pt = firedrake.Function(p.function_space())

    while True:
        pt.assign(p + t * q)
        if np.min(pt.dat.data_ro[:]) > 0:
            return t
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
                 model_args={}, dirichlet_ids=[], callback=None):
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
        callback : callable, optional
            Function to call at the end of every iteration

        The state variable must be an argument of the objective functional,
        and the paramter variable must be an argument of the regularization
        functional.
        """
        self._model = model
        self._method = method
        self._callback = callback or (lambda s: None)

        # Initialize the current guess for the parameters and the solution of
        # the forward model physics with these parameters
        self._parameter_name = parameter_name
        self._p = parameter.copy(deepcopy=True)
        self._u = state.copy(deepcopy=True)

        self._model_args = dict(**model_args, dirichlet_ids=dirichlet_ids)
        args = dict(**self._model_args,
                    **{state_name: self.state, parameter_name: self.parameter})

        # Make the form compiler use a reasonable number of quadrature points
        degree = model.quadrature_degree(**args)
        self._fc_params = {'quadrature_degree': degree}

        # Create the error and regularization functionals
        self._E = replace(objective, {state: self.state})
        self._R = replace(regularization, {parameter: self.parameter})
        self._dE = firedrake.derivative(self._E, self.state)
        dR = firedrake.derivative(self._R, self.parameter)
        self._J = self._E + self._R

        # Create the weak form of the forward model, the adjoint state, and
        # the derivative of the objective functional
        self._F = firedrake.derivative(model.action(**args), self.state)
        self._dF_du = firedrake.derivative(self._F, self.state)

        V = self.state.function_space()
        self._λ = firedrake.Function(V)
        dF_dp = firedrake.derivative(self._F, self.parameter)
        self._dJ = firedrake.action(firedrake.adjoint(dF_dp), self._λ) + dR

        # Create Dirichlet BCs where they apply for the adjoint solve
        rank = self._λ.ufl_element().num_sub_elements()
        zero = 0 if rank == 0 else (0,) * rank
        self._bc = firedrake.DirichletBC(V, zero, dirichlet_ids)

        # Create a search direction and a linear solver for projecting the
        # gradient of the objective back into the primal space
        self._solver_params = {'ksp_type': 'preonly', 'pc_type': 'lu'}
        Q = self.parameter.function_space()
        self._q = firedrake.Function(Q)
        self._M = (firedrake.TrialFunction(Q) * firedrake.TestFunction(Q) * dx
                   + firedrake.derivative(dR, self.parameter))

        # Run a first update to get the machine in a consistent state
        self.update()

        # Call the post-iteration function for the first time
        self._callback(self)

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

    def _assemble(self, *args, **kwargs):
        return firedrake.assemble(*args, **kwargs,
                                  form_compiler_parameters=self._fc_params)

    def update(self):
        u, λ, q = self.state, self.adjoint_state, self.search_direction

        # Update the observed field for the new value of the parameters
        u.assign(self._method(self._model, **self._model_args,
                              **{self._parameter_name: self._p}))

        # Update the adjoint state for the new value of the observed field
        L = firedrake.adjoint(self._dF_du)
        firedrake.solve(L == -self._dE, λ, self._bc,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

        # Update the search direction for the new parameters, observed field,
        # and adjoint state
        dJ = self.gradient
        firedrake.solve(self._M == -dJ, q,
                        solver_parameters=self._solver_params,
                        form_compiler_parameters=self._fc_params)

    def line_search(self):
        """Perform a line search along the descent direction to get a new
        value of the parameter"""
        u, p, q = self.state, self.parameter, self.search_direction
        J = self.objective + self.regularization

        s = firedrake.Constant(0)
        p_s = p + s * q
        u_s = u.copy(deepcopy=True)

        def f(t):
            s.assign(t)
            u_s.assign(self._method(self._model, **self._model_args,
                                    **{self._parameter_name: p_s}))
            return self._assemble(replace(J, {u: u_s, p: p_s}))

        try:
            a, b, c, fa, fb, fc, num_calls = scipy.optimize.bracket(f)
            b = max(a, b, c)
        except:
            b = _find_zero_crossing(p, q)

        result = scipy.optimize.minimize_scalar(f, bounds=(0, b),
                                                method='bounded')
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
        self.update()
        self._callback(self)

    def solve(self, atol=0.0, rtol=1e-6, max_iterations=None):
        """Search for a new value of the parameters, stopping once either
        the objective functional gets below a threshold value or stops
        improving."""
        max_iterations = max_iterations or np.inf
        J_initial = np.inf

        for iteration in range(max_iterations):
            J = self._assemble(self.objective)
            if ((J_initial - J) < rtol * J_initial) or (J <= atol):
                return iteration
            J_initial = J

            self.step()

        return max_iterations

