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
import matplotlib.pyplot as plt
import rasterio
import firedrake
import icepack, icepack.plot

def test_plot_mesh():
    N = 32
    mesh = firedrake.UnitSquareMesh(N, N)
    fig, ax = plt.subplots()
    icepack.plot.triplot(mesh, axes=ax)
    assert ax.legend_ is not None


def test_plot_grid_data():
    x0 = (0, 0)
    n = 32
    dx = 1.0/n
    transform = rasterio.transform.from_origin(west=0.0, north=1.0,
                                               xsize=dx, ysize=dx)

    # Interpolate a scalar field
    array = np.array([[dx * (i + j) for j in range(n + 1)]
                      for i in range(n + 1)])
    missing = -9999.0
    array[0, 0] = missing
    array = np.flipud(array)

    memfile = rasterio.MemoryFile(ext='.tif')
    opts = {'driver': 'GTiff', 'count': 1, 'width': n, 'height': n,
            'dtype': array.dtype, 'transform': transform, 'nodata': -9999}

    with memfile.open(**opts) as dataset:
        dataset.write(array, indexes=1)
    dataset = memfile.open()

    levels = np.linspace(-0.5, 0.5, 5)
    contours = icepack.plot.contourf(dataset, levels=levels)
    assert contours is not None
    colorbar = plt.colorbar(contours)
    assert colorbar is not None


def test_plot_field():
    mesh = firedrake.UnitSquareMesh(32, 32)
    Q = firedrake.FunctionSpace(mesh, 'CG', 1)
    x, y = firedrake.SpatialCoordinate(mesh)
    u = firedrake.interpolate(x * y, Q)

    filled_contours = icepack.plot.tricontourf(u)
    assert filled_contours is not None
    colorbar = plt.colorbar(filled_contours)
    assert colorbar is not None

    contours = icepack.plot.tricontour(u)
    assert contours is not None

    colors = icepack.plot.tripcolor(u)
    assert colors is not None


def test_streamlines():
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


def test_plot_extruded_field():
    Nx, Ny = 32, 32
    mesh2d = firedrake.UnitSquareMesh(Nx, Ny)
    mesh3d = firedrake.ExtrudedMesh(mesh2d, layers=1)
    x, y, z = firedrake.SpatialCoordinate(mesh3d)

    Q = firedrake.FunctionSpace(mesh3d, family='CG', degree=2,
                                vfamily='GL', vdegree=4)
    q = firedrake.interpolate((x**2 - y**2) * (1 - z**4), Q)
    q_contours = icepack.plot.tricontourf(q)
    assert q_contours is not None

    V = firedrake.VectorFunctionSpace(mesh3d, dim=2, family='CG', degree=2,
                                      vfamily='GL', vdegree=4)
    u = firedrake.interpolate(firedrake.as_vector((1 - z**4, 0)), V)
    u_contours = icepack.plot.tricontourf(u)
    assert u_contours is not None
