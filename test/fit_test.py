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
    x = firedrake.SpatialCoordinate(mesh)
    element = firedrake.FiniteElement("CG", "triangle", 1)
    Q = firedrake.FunctionSpace(mesh, element)
    p_true = firedrake.Function(Q).interpolate(true_field(x))

    # Create the point cloud and the synthetic observations. The observations
    # are polluted with a small amount of measurement noise.
    rng = np.random.default_rng(seed=1729)
    sampler = PoissonDisk(2, radius=0.1, rng=rng)
    points = sampler.fill_space()
    true_values = true_field(points)
    stddev = 1e-2
    measured_values = true_values + rng.normal(0, stddev, true_values.shape)

    # As a baseline, compute a linear fit of the measurements.
    X = np.column_stack((np.ones(len(points)), points))
    β, residual, rank, singular_values = np.linalg.lstsq(X, measured_values)
    β_0 = Constant(β[0])
    γ = Constant(β[1:])

    # Create a point cloud embedded in the unstructured mesh and define the
    # measurements on this point cloud.
    point_cloud = firedrake.VertexOnlyMesh(mesh, points, reorder=False)
    D = firedrake.FunctionSpace(point_cloud, "DG", 0)
    p_obs = firedrake.Function(D)
    p_obs.dat.data[:] = measured_values

    # Fit the observational data and check that it does an ok job at fitting.
    length = 1 / 32
    p = icepack.fit(p_obs, Constant(stddev), length, Q)

    diff = firedrake.assemble(inner(grad(p - p_true), grad(p - p_true)) * dx)
    scale = firedrake.assemble(inner(grad(p_true), grad(p_true)) * dx)
    print(f"Relative misfit in H1 norm: {diff / scale:.3f}")
    assert diff / scale < 1.0

    # In the limit as the smoothing length gets larger, we should reproduce the
    # linear fit.
    p_plane = firedrake.Function(Q).interpolate(β_0 + inner(γ, x))
    lengths = np.logspace(-6, +7, 13, base=2)
    ps = [icepack.fit(p_obs, Constant(stddev), length, Q) for length in lengths]
    diffs = np.array([firedrake.norm(p - p_plane) for p in ps])
    slope, intercept = np.polyfit(np.log(lengths), np.log(diffs), 1)
    print(f"log |p - p_plane| ~= {intercept:0.2f} {slope:+0.2f} * log α")
    assert slope < -0.8
