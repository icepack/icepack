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
from firedrake import inner, sqrt
from icepack.constants import weertman_sliding_law as m
from icepack import utilities


def tau(u, C):
    r"""Compute the shear stress for a given sliding velocity"""
    return -C * sqrt(inner(u, u))**(1/m - 1) * u


def bed_friction(u, C):
    r"""Return the bed friction part of the ice stream action functional

    The frictional part of the ice stream action functional is

    .. math::
       E(u) = -\frac{m}{m + 1}\int_\Omega\tau(u, C)\cdot u\; dx

    where :math:`\\tau(u, C)` is the basal shear stress

    .. math::
       \tau(u, C) = -C|u|^{1/m - 1}u
    """
    return -m/(m + 1) * inner(tau(u, C), u)


def side_friction(u, h, Cs=firedrake.Constant(0)):
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
    mesh = u.ufl_domain()
    ν = firedrake.FacetNormal(mesh)
    u_t = u - inner(u, ν) * ν
    return -m/(m + 1) * h * inner(tau(u_t, Cs), u_t)


def normal_flow_penalty(u, scale=1.0, exponent=None):
    r"""Return the penalty for flow normal to the domain boundary

    For problems where a glacier flows along some boundary, e.g. a fjord
    wall, the velocity has to be parallel to this boundary. Rather than
    enforce this boundary condition directly, we add a penalty for normal
    flow to the action functional.
    """
    mesh = u.ufl_domain()
    ν = firedrake.FacetNormal(mesh)
    L = utilities.diameter(mesh)
    δx = firedrake.CellSize(mesh)
    d = u.ufl_function_space().ufl_element().degree()
    exponent = d + 1 if exponent is None else exponent
    penalty = scale * (L / δx)**exponent
    return 0.5 * penalty * inner(u, ν)**2
