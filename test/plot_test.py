
import numpy as np
import firedrake
import icepack
from icepack.plot import streamline

def test_plot_mesh():
    N = 32
    mesh = firedrake.UnitSquareMesh(N, N)
    axes = icepack.plot(mesh)
    assert not (axes.legend_ is None)


def test_streamline_finite_element_field():
    N = 32
    mesh = firedrake.UnitSquareMesh(N, N)
    V = firedrake.VectorFunctionSpace(mesh, 'CG', 1)

    x, y = mesh.coordinates
    v = firedrake.interpolate(firedrake.as_vector((-y, x)), V)

    resolution = 1 / N
    radius = 0.5
    x0 = (radius, 0)
    xs = streamline(v, x0, resolution)

    num_points, _ = xs.shape
    assert num_points > 1

    for n in range(num_points):
        x = xs[n, :]
        assert abs(sum(x**2) - radius**2) < resolution


def test_streamline_grid_data():
    Nx, Ny = 32, 32
    x = np.linspace(0, 1, Nx + 1)
    y = np.linspace(0, 1, Ny + 1)
    dx, dy = 1 / Nx, 1 / Ny

    data_vx = np.zeros((Ny + 1, Nx + 1))
    data_vy = np.zeros((Ny + 1, Nx + 1))

    for i in range(Ny + 1):
        Y = i * dy
        for j in range(Nx + 1):
            X = j * dx
            data_vx[i, j] = -Y
            data_vy[i, j] = X

    from icepack.grid import GridData
    vx = GridData(x, y, data_vx, missing_data_value=np.nan)
    vy = GridData(x, y, data_vy, missing_data_value=np.nan)

    resolution = min(dx, dy)
    radius = 0.5
    x0 = (radius, 0)
    xs = streamline((vx, vy), x0, resolution)

    num_points, _ = xs.shape
    assert num_points > 1

    for n in range(num_points):
        z = xs[n, :]
        assert abs(sum(z**2) - radius**2) < resolution

