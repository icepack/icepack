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

import copy
import collections
import numpy as np
import firedrake
from firedrake import assemble
import pyadjoint


class StatisticsProblem:
    def __init__(self, simulation, loss_functional, regularization, controls):
        r"""Initialize a data assimilation problem

        Parameters
        ----------
        simulation
            A function that takes in the controls and returns the observables
        loss_functional
            A function that takes in the observables returns a form that can be
            assembled to produce the value of the model-data misfit, i.e. the
            negative logarithm of the likelihood function
        regularization
            A function that takes in the controls and returns a form that can
            be assembled to produce the value of the regularization, i.e. the
            negative log prior
        controls
            An initial guess for the unknown state or parameters; can be either
            a Firedrake Function or Constant, or an iterable of Functions or
            Constants
        """
        self._simulation = simulation
        if isinstance(loss_functional, collections.abc.Iterable):
            self._loss_functionals = list(loss_functional)
        else:
            self._loss_functionals = [loss_functional]

        if isinstance(regularization, collections.abc.Iterable):
            self._regularizations = list(regularization)
        else:
            self._regularizations = [regularization]

        if isinstance(controls, (firedrake.Function, firedrake.Constant)):
            self._controls = controls.copy(deepcopy=True)
        else:
            self._controls = [field.copy(deepcopy=True) for field in controls]

    @property
    def simulation(self):
        return self._simulation

    @property
    def loss_functional(self):
        return self._loss_functionals

    @property
    def regularization(self):
        return self._regularizations

    @property
    def controls(self):
        return self._controls


try:
    import pyadjoint.optimization.rol_solver

    class _ROLObjectiveWrapper(pyadjoint.optimization.rol_solver.ROLObjective):
        def __init__(self, *args, **kwargs):
            r"""Wrapper around the ROL objective functional class that returns
            infinity if the forward solver crashes"""
            super(_ROLObjectiveWrapper, self).__init__(*args, **kwargs)

        def update(self, x, flag, iteration):
            try:
                super().update(x, flag, iteration)
            except firedrake.ConvergenceError:
                # TODO: Remove this when we fully switch to the new pyadjoint
                # interface, the `.val` member has changed to `._val`
                if hasattr(self, "_val"):
                    self._val = np.inf
                else:
                    self.val = np.inf

    class _ROLSolverWrapper(pyadjoint.ROLSolver):
        def __init__(self, problem, controls, inner_product="L2"):
            r"""Wrapper around the ROL solver class that uses the patched Objective
            class"""
            super(_ROLSolverWrapper, self).__init__(
                problem, controls, inner_product=inner_product
            )
            self.rolobjective = _ROLObjectiveWrapper(problem.reduced_functional)

    has_rol = True
except AttributeError:
    has_rol = False


_default_rol_options = {
    "Step": {
        "Type": "Trust Region",
        "Trust Region": {
            "Initial Radius": -1,
            "Subproblem Solver": "Truncated CG",
            "Radius Growing Rate": 2.5,
            "Step Acceptance Threshold": 0.05,
            "Radius Shrinking Threshold": 0.05,
            "Radius Growing Threshold": 0.9,
            "Radius Shrinking Rate (Negative rho)": 0.0625,
            "Radius Shrinking Rate (Positive rho)": 0.25,
            "Sufficient Decrease Parameter": 1e-4,
            "Safeguard Size": 1e2,
        },
    },
    "Status Test": {
        "Gradient Tolerance": 1e-4,
        "Step Tolerance": 5e-3,
        "Iteration Limit": 50,
    },
    "General": {
        "Print Verbosity": 0,
        "Secant": {},
    },
}


class MaximumProbabilityEstimator:
    def __init__(self, problem, solver_type="rol", **kwargs):
        r"""Estimates the true value of the controls by computing the maximizer
        of the posterior probability distribution"""
        from firedrake_adjoint import Control, ReducedFunctional

        self._problem = problem
        if isinstance(problem.controls, (firedrake.Function, firedrake.Constant)):
            self._controls = problem.controls.copy(deepcopy=True)
        else:
            self._controls = [field.copy(deepcopy=True) for field in problem.controls]

        # Form the objective functional
        self._state = self.problem.simulation(self.controls)
        E = sum(assemble(E(self.state)) for E in self.problem.loss_functional)
        R = sum(assemble(R(self.controls)) for R in self.problem.regularization)
        J = E + R

        if isinstance(self.controls, (firedrake.Function, firedrake.Constant)):
            reduced_objective = ReducedFunctional(J, Control(self.controls))
        else:
            controls = [Control(field) for field in self.controls]
            reduced_objective = ReducedFunctional(J, controls)

        # Form the minimization problem and solver
        if isinstance(solver_type, str) and solver_type.lower() == "rol":
            if not has_rol:
                raise ImportError("Cannot import ROL!")

            problem_wrapper = pyadjoint.MinimizationProblem(reduced_objective)

            if "rol_options" in kwargs.keys():
                options = kwargs["rol_options"]
            else:
                options = copy.deepcopy(_default_rol_options)

                # A bunch of glue code to talk to ROL. We want to use our own
                # keyword argument names in the event that we start using a
                # different optimization package like TAO on the backend, so
                # here we're translating between our names and ROL's.
                test = options["Status Test"]
                test["Step Tolerance"] = kwargs.get("step_tolerance", 5e-3)
                test["Gradient Tolerance"] = kwargs.get("gradient_tolerance", 1e-4)
                test["Iteration Limit"] = kwargs.get("max_iterations", 50)

                general = options["General"]
                general["Print Verbosity"] = int(kwargs.get("verbose", False))

                # Use the Newton trust region algorithm if we can compute the
                # second derivative of the objective and BFGS if we can't.
                algorithm = kwargs.get("algorithm", "trust-region")
                if algorithm == "bfgs":
                    options["Step"] = {
                        "Type": "Line Search",
                        "Line Search": {
                            "Descent Method": {"Type": "Quasi-Newton Step"}
                        },
                    }
                    memory = kwargs.get("memory", 10)
                    general["Secant"] = {
                        "Type": "Limited-Memory BFGS",
                        "Maximum Storage": memory,
                    }

            self._solver = _ROLSolverWrapper(problem_wrapper, options)
        else:
            raise NotImplementedError("Only ROL solver implemented for now!")

    @property
    def problem(self):
        return self._problem

    @property
    def controls(self):
        return self._controls

    @property
    def state(self):
        return self._state

    def solve(self):
        return self._solver.solve()
