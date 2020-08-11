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
from icepack.constants import (ice_density as ρ_I, thermal_diffusivity as α,
                               heat_capacity as c, latent_heat as L,
                               melting_temperature as Tm)
from icepack.utilities import facet_normal_2, grad_2

class HeatTransport3D(object):
    r"""Class for modeling 3D heat transport

    This class solves the 3D advection-diffusion equation for the energy
    density. The energy density factors in both the temperature and the latent
    heat included in meltwater. We use the energy density rather than the
    enthalpy because it comes out to a nice round number (about 500 MPa/m^3)
    in the unit system we use.
    """
    def __init__(self, surface_exchange_coefficient=9):
        r"""Create a heat transport model

        Parameters
        ----------
        surface_exchange_coefficient : float, optional
            Penalty parameter for deviation of the surface energy from the
            atmospheric value; this is very poorly constrained
        """
        self.surface_exchange_coefficient = surface_exchange_coefficient

    def advective_flux(self, **kwargs):
        E = kwargs['E']
        u = kwargs['u']
        w = kwargs['w']
        h = kwargs['h']
        E_inflow = kwargs['E_inflow']
        E_surface = kwargs['E_surface']

        Q = E.function_space()
        ψ = firedrake.TestFunction(Q)

        U = firedrake.as_vector((u[0], u[1], w))
        flux_cells = -E * inner(U, grad(ψ)) * h * dx

        mesh = Q.mesh()
        ν = facet_normal_2(mesh)
        outflow = firedrake.max_value(inner(u, ν), 0)
        inflow = firedrake.min_value(inner(u, ν), 0)

        flux_outflow = (
            E * outflow * ψ * h * ds_v +
            E * firedrake.max_value(-w, 0) * ψ * h * ds_b +
            E * firedrake.max_value(+w, 0) * ψ * h * ds_t
        )

        flux_inflow = (
            E_inflow * inflow * ψ * h * ds_v +
            E_surface * firedrake.min_value(-w, 0) * ψ * h * ds_b +
            E_surface * firedrake.min_value(+w, 0) * ψ * h * ds_t
        )

        return flux_cells + flux_outflow + flux_inflow

    def diffusive_flux(self, **kwargs):
        E = kwargs['E']
        h = kwargs['h']
        E_surface = kwargs['E_surface']

        Q = E.function_space()
        ψ = firedrake.TestFunction(Q)

        κ = self.surface_exchange_coefficient
        cell_flux = α * E.dx(2) * ψ.dx(2) / h * dx
        surface_flux = κ * α * (E - E_surface) * ψ / h * ds_t

        return cell_flux + surface_flux

    def sources(self, **kwargs):
        E = kwargs['E']
        h = kwargs['h']
        q = kwargs['q']
        q_bed = kwargs['q_bed']

        Q = E.function_space()
        ψ = firedrake.TestFunction(Q)

        internal_sources = q * ψ * h * dx
        boundary_sources = q_bed * ψ * ds_b

        return internal_sources + boundary_sources

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
        degree_E = E.ufl_element().degree()
        degree_u = u.ufl_element().degree()
        degree = (3 * degree_E[0] + degree_u[0],
                  2 * degree_E[1] + degree_u[1])
        form_compiler_parameters = {'quadrature_degree': degree}
        firedrake.solve(F == A, E,
                        solver_parameters=solver_parameters,
                        form_compiler_parameters=form_compiler_parameters)

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

        degree_E = E.ufl_element().degree()
        degree = (3 * degree_E[0], 2 * degree_E[1])
        form_compiler_parameters = {'quadrature_degree': degree}
        firedrake.solve(a == f, E,
                        form_compiler_parameters=form_compiler_parameters)

    def solve(self, dt, E0, u, w, h, s, q, q_bed, E_inflow, E_surface):
        r"""Propagate the energy density forward by one timestep"""
        warnings.warn('Solving methods have moved to the HeatTransportSolver '
                      'class, this method will be removed in future versions.',
                      FutureWarning)

        E_inflow = E_inflow if E_inflow is not None else E0
        E_surface = E_surface if E_surface is not None else E0
        E = E0.copy(deepcopy=True)

        dt = firedrake.Constant(dt)
        self._advect(dt, E, u, w, h, s, E_inflow, E_surface)
        self._diffuse(dt, E, h, q, q_bed, E_surface)

        return E

    def temperature(self, E):
        r"""Return the temperature of ice at the given energy density"""
        return firedrake.min_value(E / (ρ_I * c), Tm)

    def meltwater_fraction(self, E):
        r"""Return the melt fraction of ice at the given energy density"""
        return firedrake.max_value(E - ρ_I * c * Tm, 0) / (ρ_I * L)

    def energy_density(self, T, f):
        r"""Return the energy density for ice at the given temperature and melt
        fraction"""
        return ρ_I * c * T + ρ_I * L * f
