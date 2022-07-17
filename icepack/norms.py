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

import numpy as np
from firedrake import sqrt, inner, grad, dx, assemble
from icepack import utilities


def norm(u, norm_type="L2"):
    r"""Compute the norm of a field

    Computes one of any number of norms of a scalar or vector field. The
    available options are:

    - ``L2``: :math:`\|u\|^2 = \int_\Omega|u|^2dx`

    - ``H01``: :math:`\|u\|^2 = \int_\Omega|\nabla u|^2dx`

    - ``H1``: :math:`\|u\|^2 = \int_\Omega\left(|u|^2 + L^2|\nabla u|^2\right)dx`

    - ``L1``: :math:`\|u\| = \int_\Omega|u|dx`

    - ``TV``: :math:`\|u\| = \int_\Omega|\nabla u|dx`

    - ``Linfty``: :math:`\|u\| = \max_{x\in\Omega}|u(x)|`

    The extra factor :math:`L` in the :math:`H^1` norm is the diameter of
    the domain in the infinity metric. This extra factor is included to
    make the norm scale appropriately with the size of the domain.
    """
    if norm_type == "L2":
        form, p = inner(u, u) * dx, 2

    if norm_type == "H01":
        form, p = inner(grad(u), grad(u)) * dx, 2

    if norm_type == "H1":
        L = utilities.diameter(u.ufl_domain())
        form, p = inner(u, u) * dx + L**2 * inner(grad(u), grad(u)) * dx, 2

    if norm_type == "L1":
        form, p = sqrt(inner(u, u)) * dx, 1

    if norm_type == "TV":
        form, p = sqrt(inner(grad(u), grad(u))) * dx, 1

    if norm_type == "Linfty":
        data = u.dat.data_ro
        if len(data.shape) == 1:
            local_max = np.max(np.abs(data))
        elif len(data.shape) == 2:
            local_max = np.max(np.sqrt(np.sum(data**2, 1)))

        return u.comm.allreduce(local_max, op=max)

    return assemble(form) ** (1 / p)
