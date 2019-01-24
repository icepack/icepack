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
describes the evolution of ice damage (levermann 2014). Next steps aim to
incorporate the Keller Hutter damage scheme used to represent 
"""

import firedrake
import icepack.models.viscosity
import math
import numpy as np
from firedrake import grad, dx, ds, dS, sqrt, Identity, inner, sym, tr as trace, det, dot, div,lt
from icepack.constants import year, glen_flow_law as n


def get_eig(hes):
    mesh = hes.function_space().mesh()
    [eigL, eigR] = np.linalg.eig(
        hes.vector().array().reshape([mesh.num_vertices(), 2, 2]))
    eig = firedrake.Function(VectorFunctionSpace(mesh, 'CG', 1))
    eig.vector().set_local(eigL.flatten())
    return eig

def M(eps, A):
    I = Identity(2)
    tr = trace(eps)
    eps_e = sqrt((inner(eps, eps) + tr**2) / 2)
    mu = 0.5 * A**(-1/n) * eps_e**(1/n - 1)
    return 2 * mu * (eps + tr * I)

def M_e(eps_e,A):
	return sqrt(3.0) * A**(-1/n) * eps_e**(1/n)

def heal(e1,eps_h,lh=2.0 * 10**-10*year):
    return lh*(e1 - eps_h)

def fracture(D,eps_e,ld=0.1):
    return ld * eps_e * (1 - D)


class DamageTransport(object):
    def solve(self, dt, D0, u, A, ld=0.1, lh=2.0 * 10**-10*year, D_inflow=None, **kwargs):
        """Propogate the ice damage forward in time by one timestep

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
        dD, ϕ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)
        d = ϕ * dD * dx
        D = D0.copy(deepcopy=True)

        """ unit normal for facets in mesh, Q """
        n = firedrake.FacetNormal(Q.mesh())

        """ find the upstream direction and solve
            for advected damage """
        un = 0.5 * (dot(u, n) + abs(dot(u, n)))
        L1 = dt * (D * div(ϕ * u) * dx
                   - firedrake.conditional(dot(u, n) < 0, ϕ * dot(u, n)
                                 * D_inflow, 0.0) * ds
                   - firedrake.conditional(dot(u, n) > 0, ϕ * dot(u, n) * D, 0.0) * ds
                   - (ϕ('+') - ϕ('-')) * (un('+') * D('+') - un('-') * D('-')) * dS)
        D1 = firedrake.Function(Q)
        D2 = firedrake.Function(Q)
        L2 = firedrake.replace(L1, {D: D1}); L3 = firedrake.replace(L1, {D: D2})

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
            for von mises criterion σc = 3.0^0.5*B*εdot**(1/n).
   			for maximum shear stress criterion (Tresca or Guest criterion) 
            σs = max(|σl|, |σt|,|σl-σt|) """   

        h_term = firedrake.Function(Q)
        f_term =firedrake.Function(Q)
        Dnew = firedrake.Function(Q)

        eps = sym(grad(u))
        tr_e = trace(eps)
        det_e = det(eps)
        eig = [1/2*tr_e + sqrt(tr_e**2 - 4*det_e), 1/2*tr_e - sqrt(tr_e**2 - 4*det_e)]
        e1 = firedrake.max_value(*eig)
        e2 = firedrake.min_value(*eig)
        eps_e = sqrt((inner(eps, eps) + tr_e**2) / 2)

        σ = M(eps,A)
        σc = M_e(eps_e,A)
        tr_s = trace(σ)
        σ_e = sqrt((inner(σ, σ) + tr_s**2) / 2)
        eps_h=2.0 * 10**-10*year
       

        """ add damage associated with longitudinal spreading after 
        advecting damage feild. Heal crevasses proportional to the  """
        h_term.project(firedrake.conditional(e1-eps_h<0,heal(e1,eps_h,lh),0.0))
        f_term.project(firedrake.conditional(σ_e - σc >0, fracture(D,eps_e,ld),0.0))

        """ we require that damage be in the set [0,1] """
        Dnew.project(firedrake.conditional(D + f_term + h_term > 1.,1.0,D + f_term + h_term ))
        Dnew.project(firedrake.conditional(D + f_term + h_term < 0.,0.0,D + f_term + h_term ))



        return Dnew
