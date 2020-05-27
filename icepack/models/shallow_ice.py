# Copyright (C) 2020 by Jessica Badgeley <badgeley@uw.edu>
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
from firedrake import inner, grad, dx
from icepack.constants import (ice_density as ρ_I, gravity as g,
                               glen_flow_law as n)
from icepack.models.mass_transport import LaxWendroff
from icepack.utilities import add_kwarg_wrapper


def mass(u):
    r"""Return mass function for the shallow ice action functional

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
    return .5 * (inner(u,u))


def gravity(u, h, s, A):
    r"""Return gravity function for the shallow ice action functional

    The gravity function for the shallow ice action functional is

    .. math::
        E(u) = \int_\Omega\frac{2A(\varrho_I g)**n}{n+2} (\nabla h^2\cdot u) h^{n+1} \nabla s^{n-1}\; dx

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness
    s : firedrake.Function
        ice surface elevation
    A : firedrake.Function or firedrake.Constant
        rate factor

    Returns
    -------
    firedrake.Form
    """
    return ((2 * A * (ρ_I * g)**n) / (n + 2)) * inner(grad(s), u) * (h ** (n + 1)) * (grad(s) ** (n - 1))


def penalty(u):
    r"""Return the penalty of the shallow ice action functional

    The penalty of the shallow ice action functional is 

    .. math::
        E(u) = \frac{1}{2}\int_\Omega l^2\nabla u\cdot \nabla u\; dx

    Parameters
    ----------
    u : firedrake.Function
        ice velocity

    Returns
    -------
    firedrake.Form
    """
    l = 2 * firedrake.CellDiameter(u.ufl_domain())
    return .5 * l**2 * inner(grad(u), grad(u))


class ShallowIce(object):
    r"""Class for modelling the flow of grounded ice
    This class provides functions that solve for the velocity, thickness,
    and surface elevation of a grounded area of slow flowing ice.

    """
    def __init__(self, mass=mass, gravity=gravity, penalty=penalty, 
                 mass_transport=LaxWendroff()):
        self.mass_transport = mass_transport
        self.mass = add_kwarg_wrapper(mass)
        self.gravity = add_kwarg_wrapper(gravity)
        self.penalty = add_kwarg_wrapper(penalty)

    def action(self, u, h, s, A, **kwargs):
        r"""Return the action functional that gives the shallow ice
        diagnostic model as the Euler-Lagrange equations

        Parameters
        ----------
        u : firedrake.Function
            Ice velocity
        h : firedrake.Function
            Ice thickness
        s : firedrake.Function
            Ice surface elevation
        A : firedrake.Function or firedrake.Constant
            Rate factor

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
        mass = self.mass(u=u, **kwargs) * dx
        gravity = self.gravity(u=u, h=h, s=s, A=A, **kwargs) * dx
        penalty = self.penalty(u=u, **kwargs) * dx
        return mass + gravity + penalty

    def scale(self, u, **kwargs):
        r"""Return the positive, convex part of the action functional

        The positive part of the action functional is used as a dimensional
        scale to determine when to terminate an optimization algorithm.
        """
        return ((self.mass(u=u, **kwargs) * dx) + (self.penalty(u=u, **kwargs) * dx))

    def diagnostic_solve(self, u0, h, s, A, **kwargs):
        r"""Solve for the ice velocity from the thickness and surface
        elevation

        Parameters
        ----------
        u0 : firedrake.Function
            Ice velocity
        h : firedrake.Function
            Ice thickness
        s : firedrake.Function
            Ice surface elevation
        A : firedrake.Function or firedrake.Constant
            Rate factor

        Returns
        -------
        u : firedrake.Function
            Ice velocity

        Other parameters
        ----------------
        **kwargs
            All other keyword arguments will be passed on to the
            'mass', 'gravity' and 'penalty' functions
            that was set when this model object was initialized
        """
        u = u0.copy(deepcopy=True)
        action = self.action(u=u, h=h, s=s, A=A, **kwargs)
                
        F = firedrake.derivative(action,u)
        firedrake.solve(F == 0, u, form_compiler_parameters={'quadrature_degree': 4})

        return u

    def prognostic_solve(self, dt, h0, a, u, h_inflow=None):
        r"""Propagate the ice thickness forward one timestep

        See :meth:`icepack.models.mass_transport.ImplicitEuler.solve`
        """
        return self.mass_transport.solve(dt, h0=h0, a=a, u=u, h_inflow=h_inflow)
