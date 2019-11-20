# Copyright (C) 2019 by Daniel Shapero <shapero@uw.edu>
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
from firedrake import norm, interpolate, Constant, as_vector, sym, grad
from icepack.utilities import eigenvalues

def test_eigenvalues():
    nx, ny = 32, 32
    mesh = firedrake.UnitSquareMesh(nx, ny)
    x, y = firedrake.SpatialCoordinate(mesh)

    V = firedrake.VectorFunctionSpace(mesh, family='CG', degree=2)
    u = interpolate(as_vector((x, 0)), V)

    Q = firedrake.FunctionSpace(mesh, family='DG', degree=2)
    ε = sym(grad(u))
    Λ1, Λ2 = eigenvalues(ε)
    λ1 = firedrake.project(Λ1, Q)
    λ2 = firedrake.project(Λ2, Q)

    assert norm(λ1 - Constant(1)) < norm(u) / (nx * ny)
    assert norm(λ2) < norm(u) / (nx * ny)
