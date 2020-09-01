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
from firedrake import inner, sqrt
from icepack.constants import weertman_sliding_law as m
from icepack.utilities import facet_normal_2, diameter, get_kwargs_alt


def friction_stress(u, C):
    r"""Compute the shear stress for a given sliding velocity"""
    return -C * sqrt(inner(u, u))**(1 / m - 1) * u


def bed_friction(u=None, C=None, **kwargs):
    r"""Return the bed friction part of the ice stream action functional

    The frictional part of the ice stream action functional is

    .. math::
       E(u) = -\frac{m}{m + 1}\int_\Omega\tau(u, C)\cdot u\; dx

    where :math:`\\tau(u, C)` is the basal shear stress

    .. math::
       \tau(u, C) = -C|u|^{1/m - 1}u
    """
    # NOTE: This mess is for backwards-compatibility, so users can still pass
    # in the velocity, thickness, and fluidity as positional arguments if they
    # are still using old code.
    if (u is not None) or (C is not None):
        warnings.warn("Abbreviated names (u, C) have been deprecated, use full"
                      " names (velocity, friction) instead.", FutureWarning)

    if u is None:
        u = kwargs['velocity']
    if C is None:
        C = kwargs['friction']

    τ = friction_stress(u, C)
    return -m / (m + 1) * inner(τ, u)


def side_friction(**kwargs):
    r"""Return the side wall friction part of the action functional

    The component of the action functional due to friction along the side
    walls of the domain is

    .. math::
       E(u) = -\frac{m}{m + 1}\int_\Gamma h\tau(u, C_s)\cdot u\; ds

    where :math:`\tau(u, C_s)` is the side wall shear stress, :math:`ds`
    is the element of surface area and :math:`\Gamma` are the side walls.
    Side wall friction is relevant for glaciers that flow through a fjord
    with rock walls on either side.
    """
    u, h = get_kwargs_alt(kwargs, ('velocity', 'thickness'), ('u', 'h'))
    Cs = kwargs.get('side_friction', kwargs.get('Cs', firedrake.Constant(0.)))

    mesh = u.ufl_domain()
    if mesh.geometric_dimension() == 2:
        ν = firedrake.FacetNormal(mesh)
    else:
        ν = facet_normal_2(mesh)

    u_t = u - inner(u, ν) * ν
    τ = friction_stress(u_t, Cs)
    return -m / (m + 1) * h * inner(τ, u_t)


def normal_flow_penalty(**kwargs):
    r"""Return the penalty for flow normal to the domain boundary

    For problems where a glacier flows along some boundary, e.g. a fjord
    wall, the velocity has to be parallel to this boundary. Rather than
    enforce this boundary condition directly, we add a penalty for normal
    flow to the action functional.
    """
    u, = get_kwargs_alt(kwargs, ('velocity',), ('u',))
    scale = kwargs.get('scale', firedrake.Constant(1.))

    mesh = u.ufl_domain()
    if mesh.geometric_dimension() == 2:
        ν = firedrake.FacetNormal(mesh)
    elif mesh.geometric_dimension() == 3:
        ν = facet_normal_2(mesh)

    L = diameter(mesh)
    δx = firedrake.FacetArea(mesh)

    # Get the polynomial degree in the horizontal direction of the velocity
    # field -- if extruded, the element degree is a tuple of the horizontal
    # and vertical degrees.
    degree = u.ufl_function_space().ufl_element().degree()
    if isinstance(degree, tuple):
        d = degree[0]
    else:
        d = degree
    exponent = kwargs.get('exponent', d + 1)

    penalty = scale * (L / δx)**exponent
    return 0.5 * penalty * inner(u, ν)**2
