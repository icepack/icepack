# Copyright (C) 2020-2021 by Jessica Badgeley <badgeley@uw.edu>
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
from icepack.constants import ice_density as ρ_I, gravity as g, glen_flow_law as n
from icepack.models.mass_transport import Continuity
from icepack.utilities import add_kwarg_wrapper


def mass(**kwargs):
    r"""Return the mass term for the shallow ice action functional

    Mass fuction for the shallow ice action functional is

    .. math::
        E(u) = \frac{1}{2}\int_\Omega u\cdot u\; dx

    Parameters
    ----------
    u : firedrake.Function
        ice velocity

    Returns
    -------
    firedrake.Form
    """
    u = kwargs["velocity"]
    return 0.5 * inner(u, u)


def gravity(**kwargs):
    r"""Return the gravity term for the shallow ice action functional

    The gravity function for the shallow ice action functional is

    .. math::
        E(u) = \int_\Omega\frac{2A(\varrho_I g)**n}{n+2} (\nabla h^2\cdot u)
        h^{n+1} \nabla s^{n-1}\; dx

    Parameters
    ----------
    velocity : firedrake.Function
    thickness : firedrake.Function
    surface : firedrake.Function
    fluidity : firedrake.Function or firedrake.Constant

    Returns
    -------
    firedrake.Form
    """
    keys = ("velocity", "thickness", "surface", "fluidity")
    u, h, s, A = itemgetter(*keys)(kwargs)

    return (
        (2 * A * (ρ_I * g) ** n / (n + 2))
        * h ** (n + 1)
        * grad(s) ** (n - 1)
        * inner(grad(s), u)
    )


def penalty(**kwargs):
    r"""Return the penalty of the shallow ice action functional

    The penalty for the shallow ice action functional is

    .. math::
        E(u) = \frac{1}{2}\int_\Omega l^2\nabla u\cdot \nabla u\; dx

    Parameters
    ----------
    velocity : firedrake.Function
    thickness : firedrake.Function

    Returns
    -------
    firedrake.Form
    """
    u, h = itemgetter("velocity", "thickness")(kwargs)
    l = 2 * firedrake.max_value(firedrake.CellDiameter(u.ufl_domain()), 5 * h)
    return 0.5 * l**2 * inner(grad(u), grad(u))


class ShallowIce:
    r"""Class for modelling the flow of grounded ice
    This class provides functions that solve for the velocity, thickness,
    and surface elevation of a grounded area of slow flowing ice.

    """

    def __init__(
        self, mass=mass, gravity=gravity, penalty=penalty, continuity=Continuity()
    ):
        self.mass = add_kwarg_wrapper(mass)
        self.gravity = add_kwarg_wrapper(gravity)
        self.penalty = add_kwarg_wrapper(penalty)
        self.continuity = continuity

    def action(self, **kwargs):
        r"""Return the action functional that gives the shallow ice
        diagnostic model as the Euler-Lagrange equations

        Parameters
        ----------
        velocity : firedrake.Function
        thickness : firedrake.Function
        surface : firedrake.Function
        fluidity : firedrake.Function or firedrake.Constant

        Returns
        -------
        E : firedrake.Form
            the shallow ice action functional

        Other parameters
        ----------------
        **kwargs
            All other keyword arguments will be passed on to the
            'mass', 'gravity' and 'penalty' functionals
        """
        metadata = {"quadrature_degree": self.quadrature_degree(**kwargs)}
        dx = firedrake.dx(metadata=metadata)
        mass = self.mass(**kwargs) * dx
        gravity = self.gravity(**kwargs) * dx
        penalty = self.penalty(**kwargs) * dx
        return mass + gravity + penalty

    def scale(self, **kwargs):
        r"""Return the positive, convex part of the action functional

        The positive part of the action functional is used as a dimensional
        scale to determine when to terminate an optimization algorithm.
        """
        metadata = {"quadrature_degree": self.quadrature_degree(**kwargs)}
        dx = firedrake.dx(metadata=metadata)
        return (self.mass(**kwargs) + self.penalty(**kwargs)) * dx

    def quadrature_degree(self, **kwargs):
        r"""Return the quadrature degree necessary to integrate the action
        functional accurately"""
        u, h = itemgetter("velocity", "thickness")(kwargs)
        degree_u = u.ufl_element().degree()
        degree_h = h.ufl_element().degree()
        return int((2 * n + 1) * degree_h + degree_u)
