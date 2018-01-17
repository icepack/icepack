
import numpy as np
import numpy.ma as ma
import pytest
from icepack.grid import GridData

def test_manual_construction():
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    data = np.array([[i + 3 * j for j in range(3)] for i in range(3)])

    # Create a gridded data set by specifying a missing data value
    missing = -9999.0
    data0 = np.copy(data)
    data0[0, 0] = missing
    dataset0 = GridData(x, y, data0, missing_data_value=missing)

    # Create a gridded data set by passing the missing data mask
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    dataset1 = GridData(x, y, data, mask=mask)

    #  Create a gridded data set by directly passing a numpy masked array
    dataset2 = GridData(x, y, ma.MaskedArray(data=data, mask=mask))

    for dataset in [dataset0, dataset1, dataset2]:
        for z in [(0.5, 1.5), (1.5, 0.5), (1.5, 1.5), (0.5, 2.0)]:
            assert abs(dataset(z) - (3*z[0] + z[1])) < 1e-6

        assert dataset.is_masked((0.5, 0.5))

        z = (2.5, 0.5)
        with pytest.raises(ValueError):
            value = dataset(z)


def test_accessing_missing_data():
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    missing = -9999.0
    data = np.array([[0, 1, 2],
                     [1, 2, 3],
                     [2, 3, missing]])
    dataset = GridData(x, y, data, missing_data_value=missing)

    assert abs(dataset((0.5, 0.5)) - 1) < 1e-6

    z = (1.5, 1.5)
    assert dataset.is_masked(z)
    with pytest.raises(ValueError):
        value = dataset(z)


def test_subset():
    N = 4
    x = np.array(range(N))
    y = np.array(range(N))
    data = np.array([[i + N * j for j in range(N)] for i in range(N)])

    dataset = GridData(x, y, data, missing_data_value=-9999.0)

    def check_subset(subset, xmin, ymin, xmax, ymax):
        assert subset.x[0] <= max(xmin, dataset.x[0])
        assert subset.y[0] <= max(ymin, dataset.y[0])
        assert subset.x[-1] >= min(xmax, dataset.x[-1])
        assert subset.y[-1] >= min(ymax, dataset.y[-1])

    xmin, ymin = -0.5, 0.5
    xmax, ymax = 1.5, 2.5
    subset = dataset.subset(xmin, ymin, xmax, ymax)
    check_subset(subset, xmin, ymin, xmax, ymax)
    assert subset.data.shape == (4, 3)

    xmin, ymin = 1.5, 2.5
    xmax, ymax = 3.5, 3.5
    subset = dataset.subset(xmin, ymin, xmax, ymax)
    check_subset(subset, xmin, ymin, xmax, ymax)
    assert subset.data.shape == (2, 3)


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
        value = dataset(z)

    file = StringIO()
    arcinfo.write(file, dataset, -2e9)
    file = StringIO(file.getvalue())
    new_dataset = arcinfo.read(file)
    for x in xs:
        assert abs(dataset(x) - new_dataset(x)) < 1e-6

    assert new_dataset.is_masked(z)

