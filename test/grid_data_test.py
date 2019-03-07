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
import numpy.ma as ma
import pytest
import icepack, icepack.grid

def test_manual_construction():
    x0 = (0, 0)
    dx = 1
    data = np.array([[i + 3 * j for j in range(3)] for i in range(3)])

    # Create a gridded data set by specifying a missing data value
    missing = -9999.0
    data0 = np.copy(data)
    data0[0, 0] = missing
    dataset0 = icepack.grid.GridData(x0, dx, data0, missing_data_value=missing)

    # Create a gridded data set by passing the missing data mask
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    dataset1 = icepack.grid.GridData(x0, dx, data, mask=mask)

    #  Create a gridded data set by directly passing a numpy masked array
    data2 = ma.MaskedArray(data=data, mask=mask)
    dataset2 = icepack.grid.GridData(x0, dx, data2)

    for dataset in [dataset0, dataset1, dataset2]:
        for z in [(0.5, 1.5), (1.5, 0.5), (1.5, 1.5), (0.5, 2.0)]:
            assert abs(dataset(z) - (3*z[0] + z[1])) < 1e-6

        assert dataset.is_masked((0.5, 0.5))

        z = (2.5, 0.5)
        with pytest.raises(ValueError):
            dataset(z)


def test_accessing_missing_data():
    x0 = (0, 0)
    dx = 1
    missing = -9999.0
    data = np.array([[0, 1, 2],
                     [1, 2, 3],
                     [2, 3, missing]])
    dataset = icepack.grid.GridData(x0, dx, data, missing_data_value=missing)

    assert abs(dataset((0.5, 0.5)) - 1) < 1e-6

    z = (1.5, 1.5)
    assert dataset.is_masked(z)
    with pytest.raises(ValueError):
        dataset(z)


def test_subset():
    N = 4
    x0 = (0, 0)
    dx = 1
    data = np.array([[i + N * j for j in range(N)] for i in range(N)])

    dataset = icepack.grid.GridData(x0, dx, data, missing_data_value=-9999.0)

    def check_subset(subset, xmin, xmax):
        x0_original = dataset.coordinate(0, 0)
        x0_subset = subset.coordinate(0, 0)
        assert x0_subset[0] <= max(xmin[0], x0_original[0])
        assert x0_subset[1] <= max(xmin[1], x0_original[1])

        ny, nx = dataset.shape
        x1_original = dataset.coordinate(ny - 1, nx - 1)

        ny, nx = subset.shape
        x1_subset = subset.coordinate(ny - 1, nx - 1)

        assert x1_subset[0] >= min(xmax[0], x1_original[0])
        assert x1_subset[1] >= min(xmax[1], x1_original[1])

    xmin = (-0.5, 0.5)
    xmax = (1.5, 2.5)
    subset = dataset.subset(xmin, xmax)
    check_subset(subset, xmin, xmax)
    assert subset.shape == (4, 3)

    xmin = (1.5, 2.5)
    xmax = (3.5, 3.5)
    subset = dataset.subset(xmin, xmax)
    check_subset(subset, xmin, xmax)
    assert subset.shape == (2, 3)


def test_arcinfo():
    raw_arcinfo = """ncols 3
                     nrows 4
                     xllcorner 1.0
                     yllcorner 2.0
                     cellsize 0.5
                     NODATA_value -9999.0
                     6.5 -9999.0 9.5
                     6.0 7.5 9.0
                     5.5 7.0 8.5
                     5.0 6.5 8.0"""

    from icepack.grid import arcinfo
    from io import StringIO
    file = StringIO(raw_arcinfo)
    dataset = arcinfo.read(file)

    xs = [(1.0, 2.0), (1.25, 2.25), (1.125, 2.99)]
    for x in xs:
        assert abs(dataset(x) - (3*x[0] + x[1])) < 1e-6

    z = (1.25, 3.01)
    assert dataset.is_masked(z)
    with pytest.raises(ValueError):
        dataset(z)

    file = StringIO()
    arcinfo.write(file, dataset, -2e9)
    file = StringIO(file.getvalue())
    new_dataset = arcinfo.read(file)
    for x in xs:
        assert abs(dataset(x) - new_dataset(x)) < 1e-6

    assert new_dataset.is_masked(z)

    with pytest.raises(FileNotFoundError):
        dataset = arcinfo.read("file_does_not_exist.txt")


def test_geotiff():
    x0 = (0, 0)
    dx = 1
    data = np.array([[i + 3 * j for j in range(3)] for i in range(3)],
                    dtype=np.int32)
    missing = -9999.0
    data[0, 0] = missing
    dataset1 = icepack.grid.GridData(x0, dx, data, missing_data_value=missing)

    import tempfile
    from icepack.grid import geotiff
    with tempfile.NamedTemporaryFile() as tmp:
        geotiff.write(tmp.name, dataset1)
        dataset2 = geotiff.read(tmp.name)
        assert np.array_equal(dataset1.data, dataset2.data)


def test_interpolating_to_mesh():
    import firedrake
    import icepack

    # Make the mesh the square `[1/4, 3/4] x [1/4, 3/4]`
    nx, ny = 32, 32
    mesh = firedrake.UnitSquareMesh(nx, ny)
    x, y = firedrake.SpatialCoordinate(mesh)
    Vc = mesh.coordinates.function_space()
    f = firedrake.interpolate(firedrake.as_vector((x/2 + 1/4, y/2 + 1/4)), Vc)
    mesh.coordinates.assign(f)

    # Interpolate a scalar field
    x0 = (0, 0)
    n = 32
    dx = 1.0/n
    data = np.array([[dx * (i + j) for j in range(n + 1)]
                     for i in range(n + 1)])
    missing = -9999.0
    data[0, 0] = missing
    dataset = icepack.grid.GridData(x0, dx, data, missing_data_value=missing)

    Q = firedrake.FunctionSpace(mesh, family='CG', degree=1)
    p = firedrake.interpolate(x + y, Q)
    q = icepack.interpolate(dataset, Q)

    assert firedrake.norm(p - q) / firedrake.norm(p) < 1e-6

    # Interpolate a vector field
    data_x = np.copy(data)
    data_y = np.array([[dx * (j - i) for j in range(n + 1)]
                       for i in range(n + 1)])
    data_y[-1, -1] = -9999.0
    vx = icepack.grid.GridData(x0, dx, data_x, missing_data_value=missing)
    vy = icepack.grid.GridData(x0, dx, data_y, missing_data_value=missing)

    V = firedrake.VectorFunctionSpace(mesh, family='CG', degree=1)
    u = firedrake.interpolate(firedrake.as_vector((x + y, x - y)), V)
    v = icepack.interpolate((vx, vy), V)

    assert firedrake.norm(u - v) / firedrake.norm(u) < 1e-6
