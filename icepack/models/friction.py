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
from firedrake import inner, dx, ds, sqrt
from icepack.constants import weertman_sliding_law as m
from icepack import utilities

def tau(u, C):
    """Compute the shear stress for a given sliding velocity
    """
    return -C * sqrt(inner(u, u))**(1/m - 1) * u


def bed_friction(u, C):
    """Return the bed friction part of the ice stream action functional

    The frictional part of the ice stream action functional is

    .. math::
       E(u) = -\\frac{m}{m + 1}\int_\Omega\\tau_b(u, C)\cdot u\hspace{2pt}dx

    where :math:`\\tau_b(u, C)` is the basal shear stress

    .. math::
       \\tau_b(u, C) = -C|u|^{1/m - 1}u
    """
    return -m/(m + 1) * inner(tau(u, C), u) * dx


def normal_flow_penalty(u, scale=1.0, exponent=None, side_wall_ids=()):
    """Return the penalty for flow normal to the domain boundary

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
    return 0.5 * penalty * inner(u, ν)**2 * ds(tuple(side_wall_ids))
