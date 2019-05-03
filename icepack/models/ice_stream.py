# Copyright (C) 2017-2019 by Daniel Shapero <shapero@uw.edu>
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
from firedrake import inner, grad, dx, ds
from icepack.constants import (ice_density as ρ_I, water_density as ρ_W,
                               gravity as g)
from icepack.models.viscosity import viscosity_depth_averaged as viscosity
from icepack.models.friction import (bed_friction, side_friction,
                                     normal_flow_penalty)
from icepack.models.mass_transport import MassTransport
from icepack.optimization import newton_search
from icepack.utilities import add_kwarg_wrapper


def gravity(u, h, s):
    r"""Return the gravitational part of the ice stream action functional

    The gravitational part of the ice stream action functional is

    .. math::
       E(u) = -\int_\Omega\rho_Igh\nabla s\cdot u\hspace{2pt}dx

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness
    s : firedrake.Function
        ice surface elevation
    """
    return -ρ_I * g * h * inner(grad(s), u) * dx


def terminus(u, h, s, ice_front_ids=()):
    r"""Return the terminal stress part of the ice stream action functional

    The power exerted due to stress at the ice calving terminus :math:`\Gamma`
    is

    .. math::
       E(u) = \int_\Gamma\left(\frac{1}{2}\rho_Igh^2 - \rho_Wgd^2\right)
       u\cdot \nu\hspace{2pt}ds

    where :math:`d` is the water depth at the terminus. We assume that sea
    level is at :math:`z = 0` for purposes of calculating the water depth.

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness
    s : firedrake.Function
        ice surface elevation
    ice_front_ids : list of int
        numeric IDs of the parts of the boundary corresponding to the
        calving front
    """
    from firedrake import conditional, lt
    d = conditional(lt(s - h, 0), s - h, 0)

    τ_I = ρ_I * g * h**2 / 2
    τ_W = ρ_W * g * d**2 / 2

    ν = firedrake.FacetNormal(u.ufl_domain())
    return (τ_I - τ_W) * inner(u, ν) * ds(tuple(ice_front_ids))


class IceStream(object):
    r"""Class for modelling the flow of grounded ice streams

    This class provides functions that solve for the velocity, thickness,
    and surface elevation of a grounded, fast-flowing ice stream.

    .. seealso::
       :py:func:`icepack.models.viscosity.viscosity_depth_averaged`
          Default implementation of the ice stream viscous action
    """

    def __init__(self, viscosity=viscosity, friction=bed_friction,
                 side_friction=side_friction, penalty=normal_flow_penalty,
                 gravity=gravity, terminus=terminus):
        self.mass_transport = MassTransport()
        self.viscosity = add_kwarg_wrapper(viscosity)
        self.friction = add_kwarg_wrapper(friction)
        self.side_friction = add_kwarg_wrapper(side_friction)
        self.penalty = add_kwarg_wrapper(penalty)
        self.gravity = add_kwarg_wrapper(gravity)
        self.terminus = add_kwarg_wrapper(terminus)

    def action(self, u, h, s, **kwargs):
        r"""Return the action functional that gives the ice stream
        diagnostic model as the Euler-Lagrange equations"""
        viscosity = self.viscosity(u=u, h=h, s=s, **kwargs)
        friction = self.friction(u=u, h=h, s=s, **kwargs)
        side_friction = self.side_friction(u=u, h=h, s=s, **kwargs)
        gravity = self.gravity(u=u, h=h, s=s, **kwargs)
        terminus = self.terminus(u=u, h=h, s=s, **kwargs)
        penalty = self.penalty(u=u, h=h, s=s, **kwargs)

        return (viscosity + friction + side_friction
                - gravity - terminus + penalty)

    def scale(self, u, h, s, **kwargs):
        r"""Return the positive, convex part of the action functional

        The positive part of the action functional is used as a dimensional
        scale to determine when to terminate an optimization algorithm.
        """
        return (self.viscosity(u=u, h=h, s=s, **kwargs)
                + self.friction(u=u, h=h, s=s, **kwargs))

    def quadrature_degree(self, u, h, **kwargs):
        r"""Return the quadrature degree necessary to integrate the action
        functional accurately

        Firedrake uses a very conservative algorithm for estimating the
        number of quadrature points necessary to integrate a given
        expression. By exploiting known structure of the problem, we can
        reduce the number of quadrature points while preserving accuracy.
        """
        degree_u = u.ufl_element().degree()
        degree_h = h.ufl_element().degree()
        return 3 * (degree_u - 1) + 2 * degree_h

    def diagnostic_solve(self, u0, h, s, dirichlet_ids, tol=1e-6, **kwargs):
        r"""Solve for the ice velocity from the thickness and surface
        elevation

        Parameters
        ----------
        u0 : firedrake.Function
            Initial guess for the ice velocity; the Dirichlet boundaries
            are taken from `u0`
        h : firedrake.Function
            Ice thickness
        s : firedrake.Function
            Ice surface elevation
        dirichlet_ids : list of int
            list of integer IDs denoting the parts of the boundary where
            Dirichlet boundary conditions should be applied
        tol : float
            dimensionless tolerance for when to terminate Newton's method

        Returns
        -------
        u : firedrake.Function
            Ice velocity

        Other parameters
        ----------------
        **kwargs
            All other keyword arguments will be passed on to the
            `viscosity`, `friction`, `gravity`, and `terminus` functions
            that were set when this model object was initialized
        """
        u = u0.copy(deepcopy=True)

        boundary_ids = u.ufl_domain().exterior_facets.unique_markers
        side_wall_ids = kwargs.get('side_wall_ids', [])
        kwargs['side_wall_ids'] = side_wall_ids
        kwargs['ice_front_ids'] = list(
            set(boundary_ids) - set(dirichlet_ids) - set(side_wall_ids))
        bcs = firedrake.DirichletBC(
            u.function_space(), firedrake.as_vector((0, 0)), dirichlet_ids)
        params = {'quadrature_degree': self.quadrature_degree(u, h, **kwargs)}

        action = self.action(u=u, h=h, s=s, **kwargs)
        scale = self.scale(u=u, h=h, s=s, **kwargs)
        return newton_search(action, u, bcs, tol, scale,
                             form_compiler_parameters=params)

    def prognostic_solve(self, dt, h0, a, u, **kwargs):
        r"""Propagate the ice thickness forward one timestep

        See :meth:`icepack.models.mass_transport.MassTransport.solve`
        """
        return self.mass_transport.solve(dt, h0=h0, a=a, u=u, **kwargs)

    def compute_surface(self, h, b):
        r"""Return the ice surface elevation consistent with a given
        thickness and bathymetry

        If the bathymetry beneath a tidewater glacier is too low, the ice
        will go afloat. The surface elevation of a floating ice shelf is

        .. math::
           s = (1 - \rho_I / \rho_W)h,

        provided everything is in hydrostatic balance.
        """
        Q = h.ufl_function_space()
        s_expr = firedrake.max_value(h + b, (1 - ρ_I / ρ_W) * h)
        return firedrake.interpolate(s_expr, Q)
