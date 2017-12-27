import math
import numpy as np
from firedrake import RectangleMesh, FunctionSpace, VectorFunctionSpace
import icepack
from icepack.models.mass_transport import MassTransport

# Test solving the mass transport equations with a constant velocity field
# and check that the solutions converge to the exact solution obtained from
# the method of characteristics.
def test_mass_transport_solver_convergence():
    L, W = 1.0, 1.0
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


# Test solving the coupled diagnostic/prognostic solvers with thickness and
# velocity fields that are exactly insteady state.
def test_coupled_diagnostic_prognostic_solver():
    from icepack.constants import rho_ice, rho_water, gravity, \
        glen_flow_law as n
    from icepack.models import IceShelf
    rho = rho_ice * (1 - rho_ice/rho_water)

    L, W = 20.0e3, 20.0e3
    h0, dh = 500.0, 100.0
    u0 = 100.0

    T = 254.15
    def velocity(x):
        q = (n + 1) * (rho * gravity * h0 * u0 / 4)**n * icepack.rate_factor(T)
        return ((u0**(n + 1) + q * x[0])**(1/(n + 1)), 0)

    def thickness(x):
        return h0 * u0 / velocity(x)[0]

    ice_shelf = IceShelf()
    degree = 2
    opts = {'dirichlet_ids': [1, 3, 4], 'tol': 1e-12}
    norm = lambda h: icepack.norm(h, norm_type='L1')

    delta_x, error = [], []
    for N in range(16, 65, 4):
        mesh = RectangleMesh(N, N, L, W)
        delta_x.append(L / N)

        V = VectorFunctionSpace(mesh, 'CG', degree)
        Q = FunctionSpace(mesh, 'CG', degree)

        u = icepack.interpolate(velocity, V)
        h = icepack.interpolate(thickness, Q)
        h_init = h.copy(deepcopy=True)
        A = icepack.interpolate(lambda x: icepack.rate_factor(T), Q)
        a = icepack.interpolate(lambda x: 0.0, Q)

        final_time, dt = 1.0, 1.0/12
        num_timesteps = math.ceil(final_time / dt)

        for k in range(num_timesteps):
            u = ice_shelf.diagnostic_solve(h=h, A=A, u0=u, **opts)
            h = ice_shelf.prognostic_solve(dt, h0=h, a=a, u=u)

        error.append(norm(h - h_init) / norm(h_init))
        print(delta_x[-1], error[-1])

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print(slope, intercept)
    assert slope > degree - 0.05
