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

import warnings
import firedrake
from firedrake import inner, grad, dx, ds
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g
)
from icepack.models.viscosity import viscosity_depth_averaged as viscosity
from icepack.models.friction import (
    bed_friction, side_friction, normal_flow_penalty
)
from icepack.models.mass_transport import LaxWendroff, Continuity
from icepack.utilities import (
    add_kwarg_wrapper,
    get_kwargs_alt,
    default_solver_parameters,
    compute_surface as _compute_surface
)


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
    keys = ('velocity', 'thickness', 'surface')
    keys_alt = ('u', 'h', 's')
    u, h, s = get_kwargs_alt(kwargs, keys, keys_alt)

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
    keys = ('velocity', 'thickness', 'surface')
    keys_alt = ('u', 'h', 's')
    u, h, s = get_kwargs_alt(kwargs, keys, keys_alt)

    d = firedrake.min_value(s - h, 0)
    τ_I = ρ_I * g * h**2 / 2
    τ_W = ρ_W * g * d**2 / 2

    ν = firedrake.FacetNormal(u.ufl_domain())
    return (τ_I - τ_W) * inner(u, ν)


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
                 gravity=gravity, terminus=terminus,
                 mass_transport=LaxWendroff(),
                 continuity=Continuity(dimension=2)):
        self.mass_transport = mass_transport
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
        u = kwargs.get('velocity', kwargs.get('u'))
        mesh = u.ufl_domain()
        ice_front_ids = tuple(kwargs.pop('ice_front_ids', ()))
        side_wall_ids = tuple(kwargs.pop('side_wall_ids', ()))

        viscosity = self.viscosity(**kwargs) * dx
        friction = self.friction(**kwargs) * dx
        gravity = self.gravity(**kwargs) * dx

        ds_w = ds(domain=mesh, subdomain_id=side_wall_ids)
        side_friction = self.side_friction(**kwargs) * ds_w
        penalty = self.penalty(**kwargs) * ds_w

        ds_t = ds(domain=mesh, subdomain_id=ice_front_ids)
        terminus = self.terminus(**kwargs) * ds_t

        return (viscosity + friction + side_friction
                - gravity - terminus + penalty)

    def quadrature_degree(self, **kwargs):
        r"""Return the quadrature degree necessary to integrate the action
        functional accurately

        Firedrake uses a very conservative algorithm for estimating the
        number of quadrature points necessary to integrate a given
        expression. By exploiting known structure of the problem, we can
        reduce the number of quadrature points while preserving accuracy.
        """
        u = kwargs.get('velocity', kwargs.get('u'))
        h = kwargs.get('thickness', kwargs.get('h'))

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
        warnings.warn('Solving methods have moved to the FlowSolver class, '
                      'this method will be removed in future versions.',
                      FutureWarning)

        u = u0.copy(deepcopy=True)

        boundary_ids = u.ufl_domain().exterior_facets.unique_markers
        side_wall_ids = kwargs.get('side_wall_ids', [])
        kwargs['side_wall_ids'] = side_wall_ids
        kwargs['ice_front_ids'] = list(
            set(boundary_ids) - set(dirichlet_ids) - set(side_wall_ids)
        )
        V = u.function_space()
        bcs = firedrake.DirichletBC(V, u, dirichlet_ids)
        params = {'quadrature_degree': self.quadrature_degree(u=u, h=h, **kwargs)}

        F = firedrake.derivative(self.action(u=u, h=h, s=s, **kwargs), u)
        firedrake.solve(
            F == 0, u, bcs,
            form_compiler_parameters=params,
            solver_parameters=default_solver_parameters
        )
        return u

    def prognostic_solve(self, dt, h0, a, u, h_inflow=None):
        r"""Propagate the ice thickness forward one timestep

        See :meth:`icepack.models.mass_transport.ImplicitEuler.solve`
        """
        return self.mass_transport.solve(dt, h0=h0, a=a, u=u, h_inflow=h_inflow)

    def compute_surface(self, h, b):
        warnings.warn('Compute surface moved from member function of models to'
                      ' icepack module; call `icepack.compute_surface` instead'
                      ' of e.g. `ice_stream.compute_surface`',
                      FutureWarning)
        return _compute_surface(h, b)
