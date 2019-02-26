import firedrake
import icepack

def test_scalar_field():
    Nx, Ny = 16, 16
    mesh2d = firedrake.UnitSquareMesh(Nx, Ny)
    mesh3d = firedrake.ExtrudedMesh(mesh2d, layers=1)
    x, y, z = firedrake.SpatialCoordinate(mesh3d)

    Q3D = firedrake.FunctionSpace(mesh3d, family='CG', degree=2,
                                  vfamily='GL', vdegree=5)
    q3d = firedrake.interpolate((x**2 + y**2) * (1 - z**4), Q3D)
    q_avg = icepack.depth_average(q3d)

    p3d = firedrake.interpolate(x**2 + y**2, Q3D)
    p_avg = icepack.depth_average(p3d, weight=1 - z**4)

    Q2D = firedrake.FunctionSpace(mesh2d, family='CG', degree=2)
    x, y = firedrake.SpatialCoordinate(mesh2d)
    q2d = firedrake.interpolate(4 * (x**2 + y**2) / 5, Q2D)

    assert q_avg.ufl_domain() is mesh2d
    assert firedrake.norm(q_avg - q2d) / firedrake.norm(q2d) < 1 / (Nx * Ny)**2
    assert firedrake.norm(p_avg - q2d) / firedrake.norm(q2d) < 1 / (Nx * Ny)**2


def test_vector_field():
    Nx, Ny = 16, 16
    mesh2d = firedrake.UnitSquareMesh(Nx, Ny)
    mesh3d = firedrake.ExtrudedMesh(mesh2d, layers=1)
    x, y, z = firedrake.SpatialCoordinate(mesh3d)

    V3D = firedrake.VectorFunctionSpace(mesh3d, dim=2, family='CG', degree=2,
                                        vfamily='GL', vdegree=5)
    u3d = firedrake.interpolate(firedrake.as_vector((1 - z**4, 0)), V3D)
    u_avg = icepack.depth_average(u3d)

    V2D = firedrake.VectorFunctionSpace(mesh2d, family='CG', degree=2)
    x, y = firedrake.SpatialCoordinate(mesh2d)
    u2d = firedrake.interpolate(firedrake.as_vector((4/5, 0)), V2D)

    assert firedrake.norm(u_avg - u2d) / firedrake.norm(u2d) < 1 / (Nx * Ny)**2
