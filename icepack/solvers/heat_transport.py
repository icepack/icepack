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

import firedrake
from firedrake import dx, Constant
from . import utilities

class HeatTransportSolver(object):
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
            if name in self.fields.keys():
                self.fields[name].assign(field)
            else:
                self.fields[name] = utilities.copy(field)

        dt = Constant(1.)

        aflux = self.model.advective_flux(**self.fields)
        dflux = self.model.diffusive_flux(**self.fields)
        sources = self.model.sources(**self.fields)
        dE_dt = sources - aflux - dflux
        E = self.fields['E']
        h = self.fields['h']
        E_0 = E.copy(deepcopy=True)
        ψ = firedrake.TestFunction(E.function_space())
        F = (E - E_0) * ψ * h * dx - dt * dE_dt

        degree = E.ufl_element().degree()
        fc_params = {'quadrature_degree': (3 * degree[0], 2 * degree[1])}
        problem = firedrake.NonlinearVariationalProblem(
            F, E, form_compiler_parameters=fc_params
        )

        solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
        self._solver = firedrake.NonlinearVariationalSolver(
            problem, solver_parameters=solver_parameters
        )

        self._energy_old = E_0
        self._timestep = dt

    def solve(self, dt, **kwargs):
        if not hasattr(self, '_solver'):
            self._setup(**kwargs)
        else:
            for name, field in kwargs.items():
                if isinstance(field, firedrake.Function):
                    self.fields[name].assign(field)

        E = self.fields['E']
        self._energy_old.assign(E)
        self._timestep.assign(dt)
        self._solver.solve()
        return E.copy(deepcopy=True)
