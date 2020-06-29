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

import firedrake
from firedrake import inner, grad, dx, ds
from icepack.constants import (ice_density as ρ_I, water_density as ρ_W,
                               gravity as g)
from icepack.models.viscosity import viscosity_depth_averaged as viscosity
from icepack.models.friction import side_friction, normal_flow_penalty
from icepack.models.mass_transport import LaxWendroff, Continuity
from icepack.optimization import MinimizationProblem, NewtonSolver
from icepack.utilities import add_kwarg_wrapper


def gravity(u, h):
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
    ρ = ρ_I * (1 - ρ_I / ρ_W)
    return -0.5 * ρ * g * inner(grad(h**2), u)


def terminus(u, h):
    r"""Return the terminus stress part of the ice shelf action functional

    The power exerted due to stress at the calving terminus :math:`\Gamma` is

    .. math::
       E(u) = \int_\Gamma\varrho gh^2u\cdot\nu\; ds

    We assume that sea level is at :math:`z = 0` for calculating the water
    depth.
    """
    mesh = u.ufl_domain()
    ν = firedrake.FacetNormal(mesh)
    ρ = ρ_I * (1 - ρ_I / ρ_W)
    return 0.5 * ρ * g * h**2 * inner(u, ν)


class IceShelf(object):
    r"""Class for modelling the flow of floating ice shelves

    This class provides functions that solve for the velocity and
    thickness of a floating ice shelf. The relevant physics can be found
    in ch. 6 of Greve and Blatter.

    .. seealso::
       :py:func:`icepack.models.viscosity.viscosity_depth_averaged`
          Default implementation of the ice shelf viscous action
    """
    def __init__(self, viscosity=viscosity, gravity=gravity, terminus=terminus,
                 side_friction=side_friction, penalty=normal_flow_penalty,
                 mass_transport=LaxWendroff(),
                 continuity=Continuity(dimension=2)):
        self.mass_transport = mass_transport
        self.viscosity = add_kwarg_wrapper(viscosity)
        self.side_friction = add_kwarg_wrapper(side_friction)
        self.penalty = add_kwarg_wrapper(penalty)
        self.gravity = add_kwarg_wrapper(gravity)
        self.terminus = add_kwarg_wrapper(terminus)
        self.continuity = continuity

    def action(self, u, h, **kwargs):
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
        mesh = u.ufl_domain()
        ice_front_ids = tuple(kwargs.pop('ice_front_ids', ()))
        side_wall_ids = tuple(kwargs.pop('side_wall_ids', ()))

        viscosity = self.viscosity(u=u, h=h, **kwargs) * dx
        gravity = self.gravity(u=u, h=h, **kwargs) * dx

        ds_w = ds(domain=mesh, subdomain_id=side_wall_ids)
        side_friction = self.side_friction(u=u, h=h, **kwargs) * ds_w
        penalty = self.penalty(u=u, h=h, **kwargs) * ds_w

        ds_t = ds(domain=mesh, subdomain_id=ice_front_ids)
        terminus = self.terminus(u=u, h=h, **kwargs) * ds_t

        return viscosity + side_friction - gravity - terminus + penalty

    def scale(self, u, h, **kwargs):
        r"""Return the positive, convex part of the action functional

        The positive part of the action functional is used as a dimensional
        scale to determine when to terminate an optimization algorithm.
        """
        return self.viscosity(u=u, h=h, **kwargs) * dx

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

    def diagnostic_solve(self, u0, h, dirichlet_ids, tol=1e-6, **kwargs):
        r"""Solve for the ice velocity from the thickness

        Parameters
        ----------
        u0 : firedrake.Function
            Initial guess for the ice velocity; the Dirichlet boundaries
            are taken from `u0`
        h : firedrake.Function
            Ice thickness
        dirichlet_ids : list of int
            list of integer IDs denoting the parts of the boundary where
            Dirichlet conditions should be applied
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
            `viscosity` and `gravity` functions that were set when this
            model object was initialized
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

        # Solve the nonlinear optimization problem
        action = self.action(u=u, h=h, **kwargs)
        scale = self.scale(u=u, h=h, **kwargs)
        problem = MinimizationProblem(action, scale, u, bcs, params)
        solver = NewtonSolver(problem, tol)
        solver.solve()
        return u

    def prognostic_solve(self, dt, h0, a, u, h_inflow=None):
        r"""Propagate the ice thickness forward one timestep

        See :meth:`icepack.models.mass_transport.MassTransport.solve`
        """
        return self.mass_transport.solve(dt, h0=h0, a=a, u=u, h_inflow=h_inflow)
