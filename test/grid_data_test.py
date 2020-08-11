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
import rasterio
import firedrake
from firedrake import dx
import icepack

def test_interpolating_function():
    nx, ny = 32, 32
    mesh = firedrake.UnitSquareMesh(nx, ny)
    x = firedrake.SpatialCoordinate(mesh)
    Q = firedrake.FunctionSpace(mesh, family='CG', degree=2)
    q = icepack.interpolate(x[0]**2 - x[1]**2, Q)
    assert abs(firedrake.assemble(q * dx)) < 1e-6


def make_rio_dataset(array, missing=-9999.):
    ny = array.shape[0] - 1
    nx = array.shape[1] - 1
    transform = rasterio.transform.from_origin(
        west=0.0, north=1.0, xsize=1 / nx, ysize=1 / ny
    )

    memfile = rasterio.MemoryFile(ext='.tif')
    opts = {
        'driver': 'GTiff',
        'count': 1,
        'width': nx + 1,
        'height': ny + 1,
        'dtype': array.dtype,
        'transform': transform,
        'nodata': missing
    }

    with memfile.open(**opts) as dataset:
        dataset.write(array, indexes=1)
    return memfile.open()


def make_domain(nx, ny, xmin, ymin, width, height):
    mesh = firedrake.UnitSquareMesh(nx, ny, diagonal='crossed')
    x, y = firedrake.SpatialCoordinate(mesh)
    Vc = mesh.coordinates.function_space()
    expr = firedrake.as_vector((width * x + xmin, height * y + ymin))
    f = firedrake.interpolate(expr, Vc)
    mesh.coordinates.assign(f)
    return mesh


def test_interpolating_scalar_field():
    n = 32
    array = np.array([[(i + j) / n for j in range(n + 1)]
                      for i in range(n + 1)])
    missing = -9999.0
    array[0, 0] = missing
    array = np.flipud(array)
    dataset = make_rio_dataset(array, missing)

    mesh = make_domain(48, 48, xmin=1/4, ymin=1/4, width=1/2, height=1/2)
    x, y = firedrake.SpatialCoordinate(mesh)
    Q = firedrake.FunctionSpace(mesh, family='CG', degree=1)
    p = firedrake.interpolate(x + y, Q)
    q = icepack.interpolate(dataset, Q)

    assert firedrake.norm(p - q) / firedrake.norm(p) < 1e-10


def test_nearest_neighbor_interpolation():
    n = 32
    array = np.array([[(i + j) / n for j in range(n + 1)]
                      for i in range(n + 1)])
    missing = -9999.0
    array[0, 0] = missing
    array = np.flipud(array)
    dataset = make_rio_dataset(array, missing)

    mesh = make_domain(48, 48, xmin=1/4, ymin=1/4, width=1/2, height=1/2)
    x, y = firedrake.SpatialCoordinate(mesh)
    Q = firedrake.FunctionSpace(mesh, family='CG', degree=1)
    p = firedrake.interpolate(x + y, Q)
    q = icepack.interpolate(dataset, Q, method='nearest')

    relative_error = firedrake.norm(p - q) / firedrake.norm(p)
    assert (relative_error > 1e-10) and (relative_error < 1 / n)


def test_interpolating_vector_field():
    n = 32
    array_vx = np.array([[(i + j) / n for j in range(n + 1)]
                         for i in range(n + 1)])
    missing = -9999.0
    array_vx[0, 0] = missing
    array_vx = np.flipud(array_vx)

    array_vy = np.array([[(j - i) / n for j in range(n + 1)]
                         for i in range(n + 1)])
    array_vy[-1, -1] = -9999.0
    array_vy = np.flipud(array_vy)

    vx = make_rio_dataset(array_vx, missing)
    vy = make_rio_dataset(array_vy, missing)

    mesh = make_domain(48, 48, xmin=1/4, ymin=1/4, width=1/2, height=1/2)
    x, y = firedrake.SpatialCoordinate(mesh)
    V = firedrake.VectorFunctionSpace(mesh, family='CG', degree=1)
    u = firedrake.interpolate(firedrake.as_vector((x + y, x - y)), V)
    v = icepack.interpolate((vx, vy), V)

    assert firedrake.norm(u - v) / firedrake.norm(u) < 1e-10


def test_close_to_edge():
    n = 32
    array = np.array([[(i + j) / n for j in range(n + 1)]
                      for i in range(n + 1)])
    missing = -9999.0
    array = np.flipud(array)
    dataset = make_rio_dataset(array, missing)

    xmin, ymin = 1 / (2 * n), 3 / (4 * n)
    mesh = make_domain(48, 48, xmin=xmin, ymin=ymin, width=1/2, height=1/2)
    Q = firedrake.FunctionSpace(mesh, family='CG', degree=1)
    q = icepack.interpolate(dataset, Q)
