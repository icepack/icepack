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

import numpy as np
import matplotlib.pyplot as plt
import firedrake
from firedrake import interpolate, as_vector
import icepack, icepack.plot

def test_plot_mesh():
    nx, ny = 32, 32
    Lx, Ly = 1e5, 1e5
    mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly)
    fig, axes = icepack.plot.subplots()
    icepack.plot.triplot(mesh, axes=axes)
    legend = axes.legend()
    assert legend is not None


def test_plot_field():
    mesh = firedrake.UnitSquareMesh(32, 32)
    Q = firedrake.FunctionSpace(mesh, family='CG', degree=1)
    x, y = firedrake.SpatialCoordinate(mesh)
    u = interpolate(x * y, Q)

    fig, axes = icepack.plot.subplots(
        nrows=2, ncols=2, sharex=True, sharey=True
    )

    filled_contours = icepack.plot.tricontourf(u, axes=axes[0, 0])
    assert filled_contours is not None
    colorbar = plt.colorbar(filled_contours, ax=axes[0, 0])
    assert colorbar is not None

    contours = icepack.plot.tricontour(u, axes=axes[0, 1])
    assert contours is not None

    colors_flat = icepack.plot.tripcolor(u, shading='flat', axes=axes[1, 0])
    assert colors_flat is not None

    colors_gouraud = icepack.plot.tripcolor(
        u, shading='gouraud', axes=axes[1, 1]
    )
    assert colors_flat.get_array().shape != colors_gouraud.get_array().shape


def test_streamlines():
    nx, ny = 32, 32
    mesh = firedrake.UnitSquareMesh(nx, ny)
    V = firedrake.VectorFunctionSpace(mesh, family='CG', degree=1)

    x, y = firedrake.SpatialCoordinate(mesh)
    v = interpolate(as_vector((-y, x)), V)

    resolution = 1 / np.sqrt(nx * ny)
    radius = 0.5
    x0 = (radius, 0)
    xs = icepack.plot.streamline(v, x0, resolution)

    num_points, _ = xs.shape
    assert num_points > 1

    for n in range(num_points):
        x = xs[n, :]
        assert abs(sum(x**2) - radius**2) < resolution


def test_plot_vector_field():
    nx, ny = 32, 32
    mesh = firedrake.UnitSquareMesh(nx, ny)
    V = firedrake.VectorFunctionSpace(mesh, family='CG', degree=1)

    x, y = firedrake.SpatialCoordinate(mesh)
    u = interpolate(as_vector((x + 0.01, x * y * (1 - y) * (y - 0.5))), V)

    fig, axes = icepack.plot.subplots(nrows=2, sharex=True, sharey=True)

    arrows = icepack.plot.quiver(u, axes=axes[0])
    assert arrows is not None

    streamlines = icepack.plot.streamplot(
        u, density=1 / nx, precision=1 / nx, axes=axes[1]
    )
    assert streamlines is not None


def test_plot_extruded_field():
    nx, ny = 32, 32
    mesh2d = firedrake.UnitSquareMesh(nx, ny)
    mesh3d = firedrake.ExtrudedMesh(mesh2d, layers=1)
    x, y, z = firedrake.SpatialCoordinate(mesh3d)

    Q = firedrake.FunctionSpace(mesh3d, 'CG', 2, vfamily='GL', vdegree=4)
    q = interpolate((x**2 - y**2) * (1 - z**4), Q)
    q_contours = icepack.plot.tricontourf(q)
    assert q_contours is not None

    V = firedrake.VectorFunctionSpace(
        mesh3d, 'CG', 2, vfamily='GL', vdegree=4, dim=2
    )
    u = interpolate(as_vector((1 - z**4, 0)), V)
    u_contours = icepack.plot.tricontourf(u)
    assert u_contours is not None
