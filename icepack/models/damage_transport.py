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
from firedrake import (inner, grad, div, dx, ds, dS, sqrt, sym, tr as trace,
                       det, min_value, max_value, conditional)
from icepack.models.viscosity import M
from icepack.constants import year, glen_flow_law as n
from icepack.utilities import eigenvalues


def heal(e1, eps_h, lh=2e-10 * year):
    return lh * (e1 - eps_h)

def fracture(D, eps_e, ld=0.1):
    return ld * eps_e * (1 - D)


class DamageTransport(object):
    def solve(self, dt, D0, u, A, ld=0.1, lh=2.0 * 10**-10*year, D_inflow=None, **kwargs):
        """Propogate the damage forward by one timestep

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
        ld : float
            damage source coefficient
        lh : float
            damage healing coefficient
        A : firedrake.Function
            fluidity parameter
        D_inflow : firedrake.Function
            Damage of the upstream ice that advects into the domain

        Returns
        D : firedrake.Function
            Ice damage at `t + dt`
        """

        D_inflow = D_inflow if D_inflow is not None else D0
        Q = D0.function_space()
        dD, φ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)
        d = φ * dD * dx
        D = D0.copy(deepcopy=True)

        """ unit normal for facets in mesh, Q """
        n = firedrake.FacetNormal(Q.mesh())

        """ find the upstream direction and solve
            for advected damage """
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

        """ three-stage strong-stability-preserving Runge-Kutta
            (SSPRK) scheme for advecting damage """

        params = {'ksp_type': 'preonly',
                  'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
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

        """ Damage advected, solve for stress and add new damage
            for von mises criterion σc = 3.0^0.5*B*ε**(1/n).
            for maximum shear stress criterion (Tresca or Guest criterion)
            σs = max(|σl|, |σt|,|σl-σt|) """

        h_term = firedrake.Function(Q)
        f_term = firedrake.Function(Q)
        Dnew = firedrake.Function(Q)

        eps = sym(grad(u))
        e1 = eigenvalues(eps)[0]
        eps_e = sqrt((inner(eps, eps) + trace(eps)**2) / 2)

        σ = M(eps, A)
        σ_e = sqrt(inner(σ, σ) - det(σ))
        eps_h = 2e-10 * year
        σc = firedrake.Constant(0.07)

        """ add damage associated with longitudinal spreading after
        advecting damage feild. Heal crevasses proportional to the  """
        h_term.project(conditional(e1 - eps_h < 0, heal(e1, eps_h, lh), 0.0))
        f_term.project(conditional(σ_e - σc > 0, fracture(D, e1, ld), 0.0))

        """ we require that damage be in the set [0,1] """
        Dnew.project(min_value(max_value(D + dt * (f_term + h_term), 0), 1))

        return Dnew
