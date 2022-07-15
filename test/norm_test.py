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
import icepack

tolerance = 1e-6

mesh = firedrake.UnitSquareMesh(16, 16)
x, y = firedrake.SpatialCoordinate(mesh)
Q = firedrake.FunctionSpace(mesh, "CG", 2)
V = firedrake.VectorFunctionSpace(mesh, "CG", 2)


def test_scalar_field_norms():
    f = firedrake.Function(Q)
    f.interpolate(x * y)

    assert abs(icepack.norm(f, norm_type="L1") - 1 / 4) < tolerance
    assert abs(icepack.norm(f, norm_type="L2") - 1 / 3) < tolerance
    assert abs(icepack.norm(f, norm_type="Linfty") - 1) < tolerance
    assert abs(icepack.norm(f, norm_type="H01") ** 2 - 2 / 3) < tolerance


def test_vector_field_norms():
    u = firedrake.Function(V)
    u.interpolate(firedrake.as_vector((x**2 - y**2, 2 * x * y)))

    assert abs(icepack.norm(u, norm_type="Linfty") - 2) < tolerance
