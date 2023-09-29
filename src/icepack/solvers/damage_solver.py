# Copyright (C) 2018-2020 by Daniel Shapero <shapero@uw.edu> and Andrew Hoffman
# <hoffmaao@uw.edu>
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

r"""Solver for the continuum damage mechanics model"""

import firedrake
from firedrake import (
    dx,
    LinearVariationalProblem,
    LinearVariationalSolver,
    max_value,
    min_value,
)


class DamageSolver:
    def __init__(self, model):
        self._model = model
        self._fields = {}

    @property
    def model(self):
        r"""The heat transport model that this object solves"""
        return self._model

    @property
    def fields(self):
        r"""The dictionary of all fields that are part of the simulation"""
        return self._fields

    def _setup(self, **kwargs):
        for name, field in kwargs.items():
            if name in self._fields.keys():
                self._fields[name].assign(field)
            else:
                if isinstance(field, firedrake.Constant):
                    self._fields[name] = firedrake.Constant(field)
                elif isinstance(field, firedrake.Function):
                    self._fields[name] = field.copy(deepcopy=True)
                else:
                    raise TypeError(
                        "Input %s field has type %s, must be Constant or Function!"
                        % (name, type(field))
                    )

        # Create symbolic representations of the flux and sources of damage
        dt = firedrake.Constant(1.0)
        flux = self.model.flux(**self.fields)

        # Create the finite element mass matrix
        D = self.fields["damage"]
        Q = D.function_space()
        φ, ψ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)
        M = φ * ψ * dx

        L1 = -dt * flux
        D1 = firedrake.Function(Q)
        D2 = firedrake.Function(Q)
        L2 = firedrake.replace(L1, {D: D1})
        L3 = firedrake.replace(L1, {D: D2})

        dD = firedrake.Function(Q)

        parameters = {
            "solver_parameters": {
                "ksp_type": "preonly",
                "pc_type": "bjacobi",
                "sub_pc_type": "ilu",
            }
        }

        problem1 = LinearVariationalProblem(M, L1, dD)
        problem2 = LinearVariationalProblem(M, L2, dD)
        problem3 = LinearVariationalProblem(M, L3, dD)
        solver1 = LinearVariationalSolver(problem1, **parameters)
        solver2 = LinearVariationalSolver(problem2, **parameters)
        solver3 = LinearVariationalSolver(problem3, **parameters)

        self._solvers = [solver1, solver2, solver3]
        self._stages = [D1, D2]
        self._damage_change = dD
        self._timestep = dt

    def solve(self, dt, **kwargs):
        if not hasattr(self, "_solvers"):
            self._setup(**kwargs)
        else:
            for name, field in kwargs.items():
                self.fields[name].assign(field)

        δt = self._timestep
        δt.assign(dt)
        D = self.fields["damage"]

        solver1, solver2, solver3 = self._solvers
        D1, D2 = self._stages
        dD = self._damage_change

        solver1.solve()
        D1.assign(D + dD)
        solver2.solve()
        D2.assign(3 / 4 * D + 1 / 4 * (D1 + dD))
        solver3.solve()
        D.assign(1 / 3 * D + 2 / 3 * (D2 + dD))

        S = self.model.sources(**self.fields)
        D.project(min_value(max_value(D + δt * S, 0), 1))
        return D.copy(deepcopy=True)
