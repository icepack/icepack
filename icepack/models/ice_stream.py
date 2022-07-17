# Copyright (C) 2017-2020 by Daniel Shapero <shapero@uw.edu>
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
from firedrake import inner
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
from icepack.models.viscosity import viscosity_depth_averaged as viscosity
from icepack.models.friction import bed_friction, side_friction, normal_flow_penalty
from icepack.models.mass_transport import Continuity
from icepack.utilities import add_kwarg_wrapper
from icepack.calculus import grad, FacetNormal, get_mesh_axes


def gravity(**kwargs):
    r"""Return the gravitational part of the ice stream action functional

    The gravitational part of the ice stream action functional is

    .. math::
       E(u) = -\int_\Omega\rho_Igh\nabla s\cdot u\; dx

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness
    s : firedrake.Function
        ice surface elevation
    """
    u, h, s = itemgetter("velocity", "thickness", "surface")(kwargs)
    return -ρ_I * g * h * inner(grad(s), u)


def terminus(**kwargs):
    r"""Return the terminal stress part of the ice stream action functional

    The power exerted due to stress at the ice calving terminus :math:`\Gamma`
    is

    .. math::
       E(u) = \frac{1}{2}\int_\Gamma\left(\rho_Igh^2 - \rho_Wgd^2\right)
       u\cdot \nu\, ds

    where :math:`d` is the water depth at the terminus. We assume that sea
    level is at :math:`z = 0` for purposes of calculating the water depth.

    Parameters
    ----------
    velocity : firedrake.Function
    thickness : firedrake.Function
    surface : firedrake.Function
    """
    u, h, s = itemgetter("velocity", "thickness", "surface")(kwargs)

    d = firedrake.min_value(s - h, 0)
    τ_I = ρ_I * g * h**2 / 2
    τ_W = ρ_W * g * d**2 / 2

    ν = FacetNormal(u.ufl_domain())
    return (τ_I - τ_W) * inner(u, ν)


class IceStream:
    r"""Class for modelling the flow of grounded ice streams

    This class provides functions that solve for the velocity, thickness,
    and surface elevation of a grounded, fast-flowing ice stream.

    .. seealso::
       :py:func:`icepack.models.viscosity.viscosity_depth_averaged`
          Default implementation of the ice stream viscous action
    """

    def __init__(
        self,
        viscosity=viscosity,
        friction=bed_friction,
        side_friction=side_friction,
        penalty=normal_flow_penalty,
        gravity=gravity,
        terminus=terminus,
        continuity=Continuity(),
    ):
        self.viscosity = add_kwarg_wrapper(viscosity)
        self.friction = add_kwarg_wrapper(friction)
        self.side_friction = add_kwarg_wrapper(side_friction)
        self.penalty = add_kwarg_wrapper(penalty)
        self.gravity = add_kwarg_wrapper(gravity)
        self.terminus = add_kwarg_wrapper(terminus)
        self.continuity = continuity

    def action(self, **kwargs):
        r"""Return the action functional that gives the ice stream
        diagnostic model as the Euler-Lagrange equations"""
        u = kwargs["velocity"]
        mesh = u.ufl_domain()
        ice_front_ids = tuple(kwargs.pop("ice_front_ids", ()))
        side_wall_ids = tuple(kwargs.pop("side_wall_ids", ()))

        metadata = {"quadrature_degree": self.quadrature_degree(**kwargs)}
        dx = firedrake.dx(metadata=metadata)
        ds = firedrake.ds(domain=mesh, metadata=metadata)

        viscosity = self.viscosity(**kwargs) * dx
        friction = self.friction(**kwargs) * dx
        gravity = self.gravity(**kwargs) * dx

        side_friction = self.side_friction(**kwargs) * ds(side_wall_ids)
        if get_mesh_axes(mesh) == "xy":
            penalty = self.penalty(**kwargs) * ds(side_wall_ids)
        else:
            penalty = 0.0

        terminus = self.terminus(**kwargs) * ds(ice_front_ids)

        return viscosity + friction + side_friction - gravity - terminus + penalty

    def scale(self, **kwargs):
        r"""Return the positive, convex part of the action functional

        The positive part of the action functional is used as a dimensional
        scale to determine when to terminate an optimization algorithm.
        """
        metadata = {"quadrature_degree": self.quadrature_degree(**kwargs)}
        dx = firedrake.dx(metadata=metadata)
        return (self.viscosity(**kwargs) + self.friction(**kwargs)) * dx

    def quadrature_degree(self, **kwargs):
        r"""Return the quadrature degree necessary to integrate the action
        functional accurately

        Firedrake uses a very conservative algorithm for estimating the
        number of quadrature points necessary to integrate a given
        expression. By exploiting known structure of the problem, we can
        reduce the number of quadrature points while preserving accuracy.
        """
        u, h = itemgetter("velocity", "thickness")(kwargs)
        degree_u = u.ufl_element().degree()
        degree_h = h.ufl_element().degree()
        return 3 * (degree_u - 1) + 2 * degree_h
