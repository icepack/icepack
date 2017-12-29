
import numpy as np
import pytest
from icepack.grid import GridData

def test_manual_construction():
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    data = np.array([[i + 3 * j for j in range(3)] for i in range(3)])

    missing = -9999.0
    dataset = GridData(x, y, data, missing)

    for z in [(0.5, 0.5), (0.5, 1.5), (1.5, 0.5), (1.5, 1.5), (1.0, 1.0)]:
        assert abs(dataset(z) - (3*z[0] + z[1])) < 1e-6

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
    dataset = GridData(x, y, data, missing)

    assert abs(dataset((0.5, 0.5)) - 1) < 1e-6

    z = (1.5, 1.5)
    assert dataset.is_masked(z)
    with pytest.raises(ValueError):
        value = dataset(z)


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

