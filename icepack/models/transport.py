# Copyright (C) 2023 by Daniel Shapero <shapero@uw.edu>
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

r"""Generic code for transport equations and mass transport"""

from operator import itemgetter
import firedrake
from firedrake import min_value, max_value, Constant, inner, grad, div, dx, ds, dS
from icepack.calculus import grad, FacetNormal


class TransportEquation:
    def __init__(self, field_name, source_name, conservative):
        self._field_name = field_name
        self._source_name = source_name
        self._conservative = conservative

    def flux(self, **kwargs):
        keys = (self._field_name, "velocity")
        q, u = itemgetter(*keys)(kwargs)
        q_inflow = kwargs.get(f"{self._field_name}_inflow", Constant(0.0))

        if q.ufl_shape != ():
            raise NotImplementedError(
                "Transport equation only implemented for scalar problems!"
            )

        Q = q.function_space()
        φ = firedrake.TestFunction(Q)

        mesh = Q.mesh()
        n = FacetNormal(mesh)
        ds = firedrake.ds if mesh.layers is None else firedrake.ds_v

        if self._conservative:
            flux_cells = -inner(q * u, grad(φ)) * dx
        else:
            flux_cells = -q * div(u * φ) * dx

        flux_out = q * max_value(0, inner(u, n)) * φ * ds
        flux_in = q_inflow * min_value(0, inner(u, n)) * φ * ds

        if q.ufl_element().family() == "Discontinuous Lagrange":
            f = q * max_value(0, inner(u, n))
            flux_faces = (f("+") - f("-")) * (φ("+") - φ("-")) * dS
            return flux_cells + flux_faces + flux_out + flux_in

        return flux_cells + flux_out + flux_in

    def sources(self, **kwargs):
        keys = (self._field_name, self._source_name)
        q, s = itemgetter(*keys)(kwargs)
        φ = firedrake.TestFunction(q.function_space())
        return s * φ * dx


class Continuity(TransportEquation):
    r"""Describes the form of the mass continuity equation"""

    def __init__(self):
        super(Continuity, self).__init__(
            field_name="thickness", source_name="accumulation", conservative=True
        )
