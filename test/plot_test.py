
import firedrake
import icepack

def test_plot_mesh():
    N = 32
    mesh = firedrake.UnitSquareMesh(N, N)
    axes = icepack.plot(mesh)
    assert not (axes.legend_ is None)

