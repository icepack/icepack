import math
import numpy as np
from firedrake import RectangleMesh, FunctionSpace, VectorFunctionSpace
import icepack
from icepack.models.mass_transport import MassTransport

L, W = 1.0, 1.0

def test_mass_transport_solver_convergence():
    u0 = 1.0
    h_in, dh = 1.0, 0.2

    def thickness_initial(x):
        return h_in - dh * x[0] / L

    def thickness_exact(x, t):
        z = (x[0] - u0 * t, x[1])
        return h_in if z[0] < 0 else thickness_initial(z)

    delta_x, error = [], []
    mass_transport = MassTransport()
    norm = lambda v: icepack.norm(v, norm_type='L1')
    for N in range(16, 97, 4):
        mesh = RectangleMesh(N, N, L, W)
        delta_x.append(L / N)

        degree = 1
        V = VectorFunctionSpace(mesh, 'CG', degree)
        Q = FunctionSpace(mesh, 'CG', degree)

        h = icepack.interpolate(thickness_initial, Q)
        a = icepack.interpolate(lambda x: 0.0, Q)
        u = icepack.interpolate(lambda x: (u0, 0.0), V)
        T = 0.5
        num_timesteps = math.ceil(0.5 * N * u0 * T / L)
        dt = T / num_timesteps

        for k in range(num_timesteps):
            h = mass_transport.solve(dt, h0=h, a=a, u=u)

        h_exact = icepack.interpolate(lambda x: thickness_exact(x, T), Q)
        error.append(norm(h - h_exact) / norm(h_exact))

        print(delta_x[-1], error[-1])

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print(slope, intercept)

