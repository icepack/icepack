# Copyright (C) 2020 by Daniel Shapero <shapero@uw.edu>
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

from operator import itemgetter
import firedrake
from firedrake import dx, Constant
from ..utilities import default_solver_parameters


class HeatTransportSolver:
    def __init__(self, model, **kwargs):
        self._model = model
        self._fields = {}

        self._solver_parameters = kwargs.get(
            "solver_parameters", default_solver_parameters
        )

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

        dt = Constant(1.0)

        aflux = self.model.advective_flux(**self.fields)
        dflux = self.model.diffusive_flux(**self.fields)
        sources = self.model.sources(**self.fields)
        dE_dt = sources - aflux - dflux
        E, h = itemgetter("energy", "thickness")(self.fields)
        E_0 = E.copy(deepcopy=True)
        ψ = firedrake.TestFunction(E.function_space())
        F = (E - E_0) * ψ * h * dx - dt * dE_dt

        degree = E.ufl_element().degree()
        fc_params = {"quadrature_degree": (3 * degree[0], 2 * degree[1])}
        problem = firedrake.NonlinearVariationalProblem(
            F, E, form_compiler_parameters=fc_params
        )

        self._solver = firedrake.NonlinearVariationalSolver(
            problem, solver_parameters=self._solver_parameters
        )

        self._energy_old = E_0
        self._timestep = dt

    def solve(self, dt, **kwargs):
        if not hasattr(self, "_solver"):
            self._setup(**kwargs)
        else:
            for name, field in kwargs.items():
                self.fields[name].assign(field)

        E = self.fields["energy"]
        self._energy_old.assign(E)
        self._timestep.assign(dt)
        self._solver.solve()
        return E.copy(deepcopy=True)
