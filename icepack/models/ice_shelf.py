# Copyright (C) 2017-2018 by Daniel Shapero <shapero@uw.edu>
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
from icepack.constants import rho_ice, rho_water, gravity as g
from icepack.models.viscosity import viscosity_depth_averaged as viscosity
from icepack.models.friction import normal_flow_penalty as penalty
from icepack.models.mass_transport import MassTransport
from icepack.optimization import newton_search
from icepack.utilities import add_kwarg_wrapper


def gravity(u, h):
    """Return the gravitational part of the ice shelf action functional

    The gravitational part of the ice shelf action functional is

    .. math::
        E(u) = -\\frac{1}{2}\int_\Omega \\varrho g\\nabla h^2\cdot u\hspace{2pt}dx

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
    rho = rho_ice * (1 - rho_ice / rho_water)
    return -0.5 * rho * g * inner(grad(h**2), u) * dx


def terminus(u, h, ice_front_ids=()):
    """Return the terminus stress part of the ice shelf action functional

    The power exerted due to stress at the calving terminus :math:`\Gamma` is

    .. math::
       E(u) = \int_\Gamma\rho_I(1 - \rho_I/\rho_W)gh^2u\cdot\\nu\hspace{2pt}ds

    We assume that sea level is at :math:`z = 0` for calculating the water
    depth.
    """
    mesh = u.ufl_domain()
    ν = firedrake.FacetNormal(mesh)
    IDs = tuple(ice_front_ids)
    rho = rho_ice * (1 - rho_ice / rho_water)
    return 0.5 * rho * g * h**2 * inner(u, ν) * ds(IDs)


class IceShelf(object):
    """Class for modelling the flow of floating ice shelves

    This class provides functions that solve for the velocity and
    thickness of a floating ice shelf. The relevant physics can be found
    in ch. 6 of Greve and Blatter.

    .. seealso::
       :py:func:`icepack.models.viscosity.viscosity_depth_averaged`
          Default implementation of the ice shelf viscous action
    """
    def __init__(self, viscosity=viscosity, gravity=gravity, terminus=terminus,
                 penalty=penalty):
        self.mass_transport = MassTransport()
        self.viscosity = add_kwarg_wrapper(viscosity)
        self.gravity = add_kwarg_wrapper(gravity)
        self.terminus = add_kwarg_wrapper(terminus)
        self.penalty = add_kwarg_wrapper(penalty)

    def action(self, u, h, **kwargs):
        """Return the action functional that gives the ice shelf diagnostic
        model as the Euler-Lagrange equations

        The action functional for the ice shelf diagnostic model is

        .. math::
            E(u) = \int_\Omega\left(\\frac{n}{n + 1}hM:\dot\\varepsilon
            - \\frac{1}{2}\\varrho gh^2\\nabla\cdot u\\right)dx

        where :math:`u` is the velocity, :math:`h` is the ice thickness,
        :math:`\dot\\varepsilon` is the strain-rate tensor, and :math:`M` is
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
        viscosity = self.viscosity(u=u, h=h, **kwargs)
        gravity = self.gravity(u=u, h=h, **kwargs)
        terminus = self.terminus(u=u, h=h, **kwargs)
        penalty = self.penalty(u=u, h=h, **kwargs)
        return viscosity - gravity - terminus + penalty

    def diagnostic_solve(self, u0, h, dirichlet_ids, tol=1e-6, **kwargs):
        """Solve for the ice velocity from the thickness

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

        # Scale the non-dimensional convergence tolerance by the viscous power
        viscosity = self.viscosity(u=u, h=h, **kwargs)
        scale = firedrake.assemble(viscosity)
        tolerance = tol * scale

        boundary_ids = u.ufl_domain().exterior_facets.unique_markers
        kwargs['side_wall_ids'] = kwargs.get('side_wall_ids', [])
        kwargs['ice_front_ids'] = list(set(boundary_ids)
            - set(dirichlet_ids) - set(kwargs['side_wall_ids']))
        bcs = [firedrake.DirichletBC(u.function_space(), (0, 0), k)
               for k in dirichlet_ids]

        # Solve the nonlinear optimization problem
        action = self.action(u=u, h=h, **kwargs)
        return newton_search(action, u, bcs, tolerance)

    def prognostic_solve(self, dt, h0, a, u, **kwargs):
        """Propagate the ice thickness forward one timestep

        Parameters
        ----------
        dt : float
            The timestep length
        h0, a : firedrake.Function
            The initial ice thickness and the accumulation rate
        u : firedrake.Function
            The ice velocity

        Returns
        -------
        h : firedrake.Function
            The new ice thickness at `t + dt`
        """
        return self.mass_transport.solve(dt, h0=h0, a=a, u=u, **kwargs)
