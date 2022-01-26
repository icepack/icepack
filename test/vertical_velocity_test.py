# Copyright (C) 2019-2020 by Daniel Shapero <shapero@uw.edu>
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
import firedrake
import icepack


def test_vertical_velocity_xyz():
    Lx, Ly = 20e3, 20e3
    nx, ny = 48, 48

    mesh2d = firedrake.RectangleMesh(nx, ny, Lx, Ly)
    mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)
    Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="DG", vdegree=0)
    Q3D = firedrake.FunctionSpace(mesh, "DG", 2, vfamily="GL", vdegree=6)
    V = firedrake.VectorFunctionSpace(mesh, "CG", 2, dim=2, vfamily="GL", vdegree=5)

    x, y, ζ = firedrake.SpatialCoordinate(mesh)

    u_inflow = 1.0
    v_inflow = 2.0
    mu = 0.003
    mv = 0.001

    b = firedrake.interpolate(firedrake.Constant(0.0), Q)
    s = firedrake.interpolate(firedrake.Constant(1000.0), Q)
    h = firedrake.interpolate(s - b, Q)
    u = firedrake.interpolate(
        firedrake.as_vector((mu * x + u_inflow, mv * y + v_inflow)), V
    )

    m = -0.01

    def analytic_vertical_velocity(h, ζ, mu, mv, m, Q3D):
        return firedrake.interpolate(
            firedrake.Constant(m) - (firedrake.Constant(mu + mv) * h * ζ), Q3D
        )

    expr = icepack.vertical_velocity(velocity=u, thickness=h, basal_mass_balance=m)
    w = firedrake.interpolate(expr * h, Q3D)
    w_analytic = analytic_vertical_velocity(h, ζ, mu, mv, m, Q3D)

    assert np.mean(np.abs(w.dat.data - w_analytic.dat.data)) < 10e-9


def test_vertical_velocity_xz():
    Lx = 20e3
    nx = 48

    mesh1d = firedrake.IntervalMesh(nx, Lx)
    mesh = firedrake.ExtrudedMesh(mesh1d, layers=1)
    Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="DG", vdegree=0)
    Q_xz = firedrake.FunctionSpace(mesh, "DG", 2, vfamily="GL", vdegree=6)
    V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=5)

    x, ζ = firedrake.SpatialCoordinate(mesh)

    u_inflow = 1.0
    mu = 0.003

    b = firedrake.interpolate(firedrake.Constant(0.0), Q)
    s = firedrake.interpolate(firedrake.Constant(1000.0), Q)
    h = firedrake.interpolate(s - b, Q)
    u = firedrake.interpolate(u_inflow + mu * x, V)

    m = -0.01

    def analytic_vertical_velocity(h, ζ, mu, m, Q_xz):
        return firedrake.interpolate(
            firedrake.Constant(m) - (firedrake.Constant(mu) * h * ζ), Q_xz
        )

    expr = icepack.vertical_velocity(velocity=u, thickness=h, basal_mass_balance=m)
    w = firedrake.interpolate(expr * h, Q_xz)
    w_analytic = analytic_vertical_velocity(h, ζ, mu, m, Q_xz)

    assert np.mean(np.abs(w.dat.data - w_analytic.dat.data)) < 10e-9
