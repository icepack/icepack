# Copyright (C) 2018-2019 by Daniel Shapero <shapero@uw.edu> and Andrew Hoffman <hoffmaao@uw.edu>
#
# This file is part of Andrew Hoffman's development branch of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

"""Solver for the damage advection equation

This module contains a solver for the conservative advection equation that
describes the evolution of ice damage (Albrecht and Levermann 2014).
"""

import numpy as np
import firedrake
from firedrake import (inner, grad, div, dx, ds, dS, sqrt, sym,
                       det, min_value, max_value, conditional)
from icepack.models.viscosity import M
from icepack.constants import year
from icepack.utilities import eigenvalues


class DamageTransport(object):
    def __init__(self, damage_stress=.07, damage_rate=.3,
                 healing_strain_rate=2e-10 * year, healing_rate=.1):
        self.damage_stress = damage_stress
        self.damage_rate = damage_rate
        self.healing_strain_rate = healing_strain_rate
        self.healing_rate = healing_rate

    def solve(self, dt, D0, u, A, D_inflow=None, **kwargs):
        r"""Propogate the damage forward by one timestep

        This function uses a Runge-Kutta scheme to upwind damage
        (limiting damage diffusion) while sourcing and sinking
        damage assocaited with crevasse opening/crevasse healing

        Parameters
        ----------
        dt : float
            Timestep
        D0 : firedrake.Function
            initial damage feild should be discontinuous
        u : firedrake.Function
            Ice velocity
        A : firedrake.Function
            fluidity parameter
        D_inflow : firedrake.Function
            Damage of the upstream ice that advects into the domain

        Returns
        -------
        D : firedrake.Function
            Ice damage at `t + dt`
        """

        D_inflow = D_inflow if D_inflow is not None else D0
        Q = D0.function_space()
        dD, φ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)
        d = φ * dD * dx
        D = D0.copy(deepcopy=True)

        n = firedrake.FacetNormal(Q.mesh())

        un = 0.5 * (inner(u, n) + abs(inner(u, n)))
        L1 = dt * (D * div(φ * u) * dx
                   - φ * max_value(inner(u, n), 0) * D * ds
                   - φ * min_value(inner(u, n), 0) * D_inflow * ds
                   - (φ('+') - φ('-')) * (un('+') * D('+') - un('-') * D('-')) * dS)
        D1 = firedrake.Function(Q)
        D2 = firedrake.Function(Q)
        L2 = firedrake.replace(L1, {D: D1})
        L3 = firedrake.replace(L1, {D: D2})

        dq = firedrake.Function(Q)

        # Three-stage strong structure-preserving Runge Kutta (SSPRK3) method
        params = {
            'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'
        }
        prob1 = firedrake.LinearVariationalProblem(d, L1, dq)
        solv1 = firedrake.LinearVariationalSolver(prob1, solver_parameters=params)
        prob2 = firedrake.LinearVariationalProblem(d, L2, dq)
        solv2 = firedrake.LinearVariationalSolver(prob2, solver_parameters=params)
        prob3 = firedrake.LinearVariationalProblem(d, L3, dq)
        solv3 = firedrake.LinearVariationalSolver(prob3, solver_parameters=params)

        solv1.solve()
        D1.assign(D + dq)
        solv2.solve()
        D2.assign(0.75 * D + 0.25 * (D1 + dq))
        solv3.solve()
        D.assign((1.0 / 3.0) * D + (2.0 / 3.0) * (D2 + dq))

        # Increase/decrease damage depending on stress and strain rates
        ε = sym(grad(u))
        ε_1 = eigenvalues(ε)[0]

        σ = M(ε, A)
        σ_e = sqrt(inner(σ, σ) - det(σ))

        ε_h = firedrake.Constant(self.healing_strain_rate)
        σ_d = firedrake.Constant(self.damage_stress)
        γ_h = firedrake.Constant(self.healing_rate)
        γ_d = firedrake.Constant(self.damage_rate)

        healing = γ_h * min_value(ε_1 - ε_h, 0)
        fracture = γ_d * conditional(σ_e - σ_d > 0, ε_1, 0.) * (1 - D)

        # Clamp damage field to [0, 1]
        D.project(min_value(max_value(D + dt * (healing + fracture), 0), 1))
        return D
