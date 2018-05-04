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

