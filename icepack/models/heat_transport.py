# Copyright (C) 2019 by Daniel Shapero <shapero@uw.edu>
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
from firedrake import inner, grad, div, dx, ds, ds_b, ds_t, ds_v
from icepack.constants import rho_ice as ρ_I, thermal_diffusivity as α
from icepack.utilities import facet_normal_2, grad_2

class HeatTransport3D(object):
    r"""Class for modeling 3D heat transport

    This class solves the 3D advection-diffusion equation for the energy
    density. The energy density factors in both the temperature and the latent
    heat included in meltwater. We use the energy density rather than the
    enthalpy because it comes out to a nice round number (about 500 MPa/m^3)
    in the unit system we use.
    """
    def __init__(self):
        pass

    def _advect(self, dt, E, u, w, h, s, E_inflow, E_surface):
        Q = E.function_space()
        φ, ψ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)

        U = firedrake.as_vector((u[0], u[1], w))
        flux_cells = -φ * inner(U, grad(ψ)) * h * dx

        mesh = Q.mesh()
        ν = facet_normal_2(mesh)
        outflow = firedrake.max_value(inner(u, ν), 0)
        inflow = firedrake.min_value(inner(u, ν), 0)

        flux_outflow = φ * ψ * outflow * h * ds_v + \
                       φ * ψ * firedrake.max_value(-w, 0) * h * ds_b + \
                       φ * ψ * firedrake.max_value(+w, 0) * h * ds_t
        F = φ * ψ * h * dx + dt * (flux_cells + flux_outflow)

        flux_inflow = -E_inflow * ψ * inflow * h * ds_v \
                      -E_surface * ψ * firedrake.min_value(-w, 0) * h * ds_b \
                      -E_surface * ψ * firedrake.min_value(+w, 0) * h * ds_t
        A = E * ψ * h * dx + dt * flux_inflow

        solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
        firedrake.solve(F == A, E, solver_parameters=solver_parameters)

    def _diffuse(self, dt, E, h, q, q_bed, E_surface):
        Q = E.function_space()
        degree = Q.ufl_element().degree()[1]
        φ, ψ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)

        a = (h * φ * ψ + dt * α * φ.dx(2) * ψ.dx(2) / h) * dx \
            + degree**2 * dt * α * φ * ψ / h * ds_t
        f = E * ψ * h * dx \
            + dt * q * ψ * h * dx \
            + dt * q_bed * ψ * ds_b \
            + degree**2 * dt * α * E_surface * ψ / h * ds_t
        firedrake.solve(a == f, E)

    def solve(self, dt, E0, u, w, h, s, q, q_bed, E_inflow, E_surface):
        r"""Propagate the energy density forward by one timestep"""
        E_inflow = E_inflow if E_inflow is not None else E0
        E_surface = E_surface if E_surface is not None else E0
        E = E0.copy(deepcopy=True)

        dt = firedrake.Constant(dt)
        self._advect(dt, E, u, w, h, s, E_inflow, E_surface)
        self._diffuse(dt, E, h, q, q_bed, E_surface)

        return E
