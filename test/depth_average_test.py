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
from firedrake import norm
from icepack import depth_average, lift3d

def test_scalar_field():
    Nx, Ny = 16, 16
    mesh2d = firedrake.UnitSquareMesh(Nx, Ny)
    mesh3d = firedrake.ExtrudedMesh(mesh2d, layers=1)
    x, y, z = firedrake.SpatialCoordinate(mesh3d)

    Q3D = firedrake.FunctionSpace(mesh3d, family='CG', degree=2,
                                  vfamily='GL', vdegree=5)
    q3d = firedrake.interpolate((x**2 + y**2) * (1 - z**4), Q3D)
    q_avg = depth_average(q3d)

    p3d = firedrake.interpolate(x**2 + y**2, Q3D)
    p_avg = depth_average(p3d, weight=1 - z**4)

    Q2D = firedrake.FunctionSpace(mesh2d, family='CG', degree=2)
    x, y = firedrake.SpatialCoordinate(mesh2d)
    q2d = firedrake.interpolate(4 * (x**2 + y**2) / 5, Q2D)

    assert q_avg.ufl_domain() is mesh2d
    assert norm(q_avg - q2d) / norm(q2d) < 1 / (Nx * Ny)**2
    assert norm(p_avg - q2d) / norm(q2d) < 1 / (Nx * Ny)**2

    Q0 = firedrake.FunctionSpace(mesh3d, family='CG', degree=2,
                                 vfamily='GL', vdegree=0)
    q_lift = lift3d(q_avg, Q0)
    assert norm(depth_average(q_lift) - q2d) / norm(q2d) < 1 / (Nx * Ny)**2


def test_vector_field():
    Nx, Ny = 16, 16
    mesh2d = firedrake.UnitSquareMesh(Nx, Ny)
    mesh3d = firedrake.ExtrudedMesh(mesh2d, layers=1)
    x, y, z = firedrake.SpatialCoordinate(mesh3d)

    V3D = firedrake.VectorFunctionSpace(mesh3d, dim=2, family='CG', degree=2,
                                        vfamily='GL', vdegree=5)
    u3d = firedrake.interpolate(firedrake.as_vector((1 - z**4, 0)), V3D)
    u_avg = depth_average(u3d)

    V2D = firedrake.VectorFunctionSpace(mesh2d, family='CG', degree=2)
    x, y = firedrake.SpatialCoordinate(mesh2d)
    u2d = firedrake.interpolate(firedrake.as_vector((4/5, 0)), V2D)

    assert norm(u_avg - u2d) / norm(u2d) < 1 / (Nx * Ny)**2

    V0 = firedrake.VectorFunctionSpace(mesh3d, dim=2, family='CG', degree=2,
                                       vfamily='GL', vdegree=0)
    u_lift = lift3d(u_avg, V0)
    assert norm(depth_average(u_lift) - u2d) / norm(u2d) < 1 / (Nx * Ny)**2
