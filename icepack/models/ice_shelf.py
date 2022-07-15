# Copyright (C) 2017-2021 by Daniel Shapero <shapero@uw.edu>
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
from firedrake import inner, grad
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
from icepack.models.viscosity import viscosity_depth_averaged as viscosity
from icepack.models.friction import side_friction, normal_flow_penalty
from icepack.models.mass_transport import Continuity
from icepack.utilities import add_kwarg_wrapper


def gravity(**kwargs):
    r"""Return the gravitational part of the ice shelf action functional

    The gravitational part of the ice shelf action functional is

    .. math::
        E(u) = -\frac{1}{2}\int_\Omega\varrho g\nabla h^2\cdot u\; dx

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness

    Returns
    -------
    firedrake.Form
    """
    u, h = itemgetter("velocity", "thickness")(kwargs)

    ρ = ρ_I * (1 - ρ_I / ρ_W)
    return -0.5 * ρ * g * inner(grad(h**2), u)


def terminus(**kwargs):
    r"""Return the terminus stress part of the ice shelf action functional

    The power exerted due to stress at the calving terminus :math:`\Gamma` is

    .. math::
       E(u) = \int_\Gamma\varrho gh^2u\cdot\nu\; ds

    We assume that sea level is at :math:`z = 0` for calculating the water
    depth.
    """
    u, h = itemgetter("velocity", "thickness")(kwargs)

    mesh = u.ufl_domain()
    ν = firedrake.FacetNormal(mesh)
    ρ = ρ_I * (1 - ρ_I / ρ_W)
    return 0.5 * ρ * g * h**2 * inner(u, ν)


class IceShelf:
    r"""Class for modelling the flow of floating ice shelves

    This class provides functions that solve for the velocity and
    thickness of a floating ice shelf. The relevant physics can be found
    in ch. 6 of Greve and Blatter.

    .. seealso::
       :py:func:`icepack.models.viscosity.viscosity_depth_averaged`
          Default implementation of the ice shelf viscous action
    """

    def __init__(
        self,
        viscosity=viscosity,
        gravity=gravity,
        terminus=terminus,
        side_friction=side_friction,
        penalty=normal_flow_penalty,
        continuity=Continuity(),
    ):
        self.viscosity = add_kwarg_wrapper(viscosity)
        self.side_friction = add_kwarg_wrapper(side_friction)
        self.penalty = add_kwarg_wrapper(penalty)
        self.gravity = add_kwarg_wrapper(gravity)
        self.terminus = add_kwarg_wrapper(terminus)
        self.continuity = continuity

    def action(self, **kwargs):
        r"""Return the action functional that gives the ice shelf diagnostic
        model as the Euler-Lagrange equations

        The action functional for the ice shelf diagnostic model is

        .. math::
            E(u) = \int_\Omega\left(\frac{n}{n + 1}hM:\dot\varepsilon
            - \frac{1}{2}\varrho gh^2\nabla\cdot u\right)dx

        where :math:`u` is the velocity, :math:`h` is the ice thickness,
        :math:`\dot\varepsilon` is the strain-rate tensor, and :math:`M` is
        the membrane stress tensor.

        Parameters
        ----------
        u : firedrake.Function
            ice velocity
        h : firedrake.Function
            ice thickness

        Returns
        -------
        E : firedrake.Form
            the ice shelf action functional

        Other parameters
        ----------------
        **kwargs
            All other keyword arguments will be passed on to the viscosity
            and gravity functionals. The ice fluidity coefficient, for
            example, is passed as a keyword argument.
        """
        u = kwargs["velocity"]
        mesh = u.ufl_domain()
        ice_front_ids = tuple(kwargs.pop("ice_front_ids", ()))
        side_wall_ids = tuple(kwargs.pop("side_wall_ids", ()))

        metadata = {"quadrature_degree": self.quadrature_degree(**kwargs)}
        dx = firedrake.dx(metadata=metadata)
        ds = firedrake.ds(domain=mesh, metadata=metadata)

        viscosity = self.viscosity(**kwargs) * dx
        gravity = self.gravity(**kwargs) * dx

        side_friction = self.side_friction(**kwargs) * ds(side_wall_ids)
        penalty = self.penalty(**kwargs) * ds(side_wall_ids)
        terminus = self.terminus(**kwargs) * ds(ice_front_ids)

        return viscosity + side_friction - gravity - terminus + penalty

    def scale(self, **kwargs):
        r"""Return the positive, convex part of the action functional

        The positive part of the action functional is used as a dimensional
        scale to determine when to terminate an optimization algorithm.
        """
        metadata = {"quadrature_degree": self.quadrature_degree(**kwargs)}
        dx = firedrake.dx(metadata=metadata)
        return self.viscosity(**kwargs) * dx

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
