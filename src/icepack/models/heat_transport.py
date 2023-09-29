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

from operator import itemgetter
import firedrake
from firedrake import inner, grad, dx, ds_b, ds_t, ds_v
from icepack.constants import (
    ice_density as ρ_I,
    thermal_diffusivity as α,
    heat_capacity as c,
    latent_heat as L,
    melting_temperature as Tm,
)
from icepack.calculus import FacetNormal


class HeatTransport3D:
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
        keys = (
            "energy",
            "velocity",
            "vertical_velocity",
            "thickness",
            "energy_inflow",
            "energy_surface",
        )
        E, u, w, h, E_inflow, E_surface = itemgetter(*keys)(kwargs)

        Q = E.function_space()
        ψ = firedrake.TestFunction(Q)

        # NOTE: Be careful here going to xz! You might have to separate this into
        # the sum of a horizontal and vertical flux if we're shadowing Firedrake's
        # grad operator with out own specialized one.
        U = firedrake.as_vector((u[0], u[1], w))
        flux_cells = -E * inner(U, grad(ψ)) * h * dx

        ν = FacetNormal(Q.mesh())
        outflow = firedrake.max_value(inner(u, ν), 0)
        inflow = firedrake.min_value(inner(u, ν), 0)

        flux_outflow = (
            E * outflow * ψ * h * ds_v
            + E * firedrake.max_value(-w, 0) * ψ * h * ds_b
            + E * firedrake.max_value(+w, 0) * ψ * h * ds_t
        )

        flux_inflow = (
            E_inflow * inflow * ψ * h * ds_v
            + E_surface * firedrake.min_value(-w, 0) * ψ * h * ds_b
            + E_surface * firedrake.min_value(+w, 0) * ψ * h * ds_t
        )

        return flux_cells + flux_outflow + flux_inflow

    def diffusive_flux(self, **kwargs):
        keys = ("energy", "thickness", "energy_surface")
        E, h, E_surface = itemgetter(*keys)(kwargs)

        Q = E.function_space()
        ψ = firedrake.TestFunction(Q)

        κ = self.surface_exchange_coefficient
        cell_flux = α * E.dx(2) * ψ.dx(2) / h * dx
        surface_flux = κ * α * (E - E_surface) * ψ / h * ds_t

        return cell_flux + surface_flux

    def sources(self, **kwargs):
        keys = ("energy", "thickness", "heat", "heat_bed")
        E, h, q, q_bed = itemgetter(*keys)(kwargs)

        Q = E.function_space()
        ψ = firedrake.TestFunction(Q)

        internal_sources = q * ψ * h * dx
        boundary_sources = q_bed * ψ * ds_b

        return internal_sources + boundary_sources

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
