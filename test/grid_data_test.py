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
import rasterio
import firedrake
import icepack

def test_interpolating_to_mesh():
    # Make the mesh the square `[1/4, 3/4] x [1/4, 3/4]`
    nx, ny = 32, 32
    mesh = firedrake.UnitSquareMesh(nx, ny)
    x, y = firedrake.SpatialCoordinate(mesh)
    Vc = mesh.coordinates.function_space()
    f = firedrake.interpolate(firedrake.as_vector((x/2 + 1/4, y/2 + 1/4)), Vc)
    mesh.coordinates.assign(f)

    # Set up the geometry of the gridded data set
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
    opts = {'driver': 'GTiff', 'count': 1, 'width': n + 1, 'height': n + 1,
            'dtype': array.dtype, 'transform': transform, 'nodata': -9999}

    with memfile.open(**opts) as dataset:
        dataset.write(array, indexes=1)
    dataset = memfile.open()

    Q = firedrake.FunctionSpace(mesh, family='CG', degree=1)
    p = firedrake.interpolate(x + y, Q)
    q = icepack.interpolate(dataset, Q)

    assert firedrake.norm(p - q) / firedrake.norm(p) < 1e-6

    # Interpolate a vector field
    array_vx = np.copy(array)
    array_vy = np.array([[dx * (j - i) for j in range(n + 1)]
                         for i in range(n + 1)])
    array_vy[-1, -1] = -9999.0
    array_vy = np.flipud(array_vy)

    memfile_vx, memfile_vy = rasterio.MemoryFile(), rasterio.MemoryFile()
    with memfile_vx.open(**opts) as vx, memfile_vy.open(**opts) as vy:
        vx.write(array_vx, indexes=1)
        vy.write(array_vy, indexes=1)
    vx, vy = memfile_vx.open(), memfile_vy.open()

    V = firedrake.VectorFunctionSpace(mesh, family='CG', degree=1)
    u = firedrake.interpolate(firedrake.as_vector((x + y, x - y)), V)
    v = icepack.interpolate((vx, vy), V)

    assert firedrake.norm(u - v) / firedrake.norm(u) < 1e-6
