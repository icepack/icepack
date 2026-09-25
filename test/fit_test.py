from importlib.metadata import version
from packaging.version import Version
import pytest
import numpy as np
from numpy import pi as π
from scipy.stats.qmc import PoissonDisk
import firedrake
from firedrake import Constant, inner, grad, dx
import icepack

a = 1.0
b = (-1/4, +1/4)
c = 1/8
k = (6, 2)
f = 1/5

def true_field(x):
    if isinstance(x, firedrake.SpatialCoordinate):
        A = Constant(a)
        B = Constant(b)
        C = Constant(c)
        K = Constant(k)
        F = Constant(f)
        return A + inner(B, x) + C * firedrake.cos(π * (inner(K, x) + F))

    return a + x @ np.array(b) + c * np.cos(π * (x @ np.array(k) + f))


@pytest.mark.xfail(
    Version(version("firedrake")) < Version("2026.4.2"),
    reason="Sparse data fitting only works on recent firedrake >= 2026.4.2"
)
def test_fitting_sparse_data():
    n = 32
    mesh = firedrake.UnitSquareMesh(n, n, diagonal="crossed")
    element = firedrake.FiniteElement("CG", "triangle", 1)
    Q = firedrake.FunctionSpace(mesh, element)

    # Create the point cloud and the synthetic observations. The observations
    # are polluted with a small amount of measurement noise.
    rng = np.random.default_rng(seed=1729)
    sampler = PoissonDisk(2, radius=0.1, rng=rng)
    points = sampler.fill_space()
    point_cloud = firedrake.VertexOnlyMesh(mesh, points, reorder=False)
    true_values = true_field(points)

    stddev = 1e-2
    D = firedrake.FunctionSpace(point_cloud, "DG", 0)
    p_obs = firedrake.Function(D)
    p_obs.dat.data[:] = true_values + rng.normal(0, stddev, true_values.shape)

    # Fit the observational data and compare the results.
    length = 1 / 16
    p = icepack.fit(p_obs, Constant(stddev), length, Q)

    x = firedrake.SpatialCoordinate(mesh)
    p_true = firedrake.Function(Q).interpolate(true_field(x))

    diff = firedrake.assemble(inner(grad(p - p_true), grad(p - p_true)) * dx)
    scale = firedrake.assemble(inner(grad(p_true), grad(p_true)) * dx)
    print(f"Relative misfit in H1 norm: {diff / scale:.3f}")
    assert diff / scale < 1.0
