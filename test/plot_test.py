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

import numpy as np
import matplotlib.pyplot as plt
import firedrake
import icepack, icepack.plot
from icepack.grid import GridData

def test_plot_mesh():
    N = 32
    mesh = firedrake.UnitSquareMesh(N, N)
    fig, ax = plt.subplots()
    icepack.plot.triplot(mesh, axes=ax)
    assert ax.legend_ is not None


def test_plot_grid_data():
    x0 = (0, 0)
    N = 32
    delta = 1 / N
    data = np.zeros((N + 1, N + 1))

    for i in range(N):
        y = i * delta
        for j in range(N):
            x = j * delta
            data[i, j] = (x - 0.5) * (y - 0.5)

    dataset = GridData(x0, delta, data, missing_data_value=np.nan)
    levels = [-0.5 + 0.25 * n for n in range(5)]
    contours = icepack.plot.contourf(dataset, levels=levels)
    assert contours is not None
    colorbar = plt.colorbar(contours)
    assert colorbar is not None


def test_plot_field():
    mesh = firedrake.UnitSquareMesh(32, 32)
    Q = firedrake.FunctionSpace(mesh, 'CG', 1)
    x, y = firedrake.SpatialCoordinate(mesh)
    u = firedrake.interpolate(x * y, Q)
    contours = icepack.plot.tricontourf(u)
    assert contours is not None
    colorbar = plt.colorbar(contours)
    assert colorbar is not None


def test_streamline_finite_element_field():
    N = 32
    mesh = firedrake.UnitSquareMesh(N, N)
    V = firedrake.VectorFunctionSpace(mesh, 'CG', 1)

    x, y = firedrake.SpatialCoordinate(mesh)
    v = firedrake.interpolate(firedrake.as_vector((-y, x)), V)

    resolution = 1 / N
    radius = 0.5
    x0 = (radius, 0)
    xs = icepack.plot.streamline(v, x0, resolution)

    num_points, _ = xs.shape
    assert num_points > 1

    for n in range(num_points):
        x = xs[n, :]
        assert abs(sum(x**2) - radius**2) < resolution


def test_streamline_grid_data():
    N = 32
    data_vx = np.zeros((N + 1, N + 1))
    data_vy = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        Y = i / N
        for j in range(N + 1):
            X = j / N
            data_vx[i, j] = -Y
            data_vy[i, j] = X

    vx = GridData((0, 0), 1/N, data_vx, missing_data_value=np.nan)
    vy = GridData((0, 0), 1/N, data_vy, missing_data_value=np.nan)

    radius = 0.5
    x0 = (radius, 0)
    xs = icepack.plot.streamline((vx, vy), x0, 1/N)

    num_points, _ = xs.shape
    assert num_points > 1

    for n in range(num_points):
        z = xs[n, :]
        assert abs(sum(z**2) - radius**2) < 1/N
