
import numpy as np
import firedrake
from firedrake import inner, grad, dx, assemble
import icepack, icepack.models
from icepack.constants import gravity, rho_ice, rho_water, glen_flow_law as n

# The domain is a 20km x 20km square, with ice flowing in from the left.
L, W = 20.0e3, 20.0e3

def test_diagnostic_solver_convergence():
    def norm(u):
        form = inner(u, u) * dx + L**2 * inner(grad(u), grad(u)) * dx
        norm_squared = assemble(form)
        return np.sqrt(norm_squared)

    h0, dh = 500.0, 100.0

    # This is an exact solution for the velocity of a floating ice shelf with
    # constant temperature and linearly decreasing thickness. See Greve and
    # Blatter for the derivation.
    T = 254.15
    def velocity_exact(x):
        u0 = 100.0
        rho = rho_ice * (1 - rho_ice/rho_water)
        A = icepack.rate_factor(T) * (rho * gravity * h0 / 4)**n
        q = 1 - (1 - (dh / h0) * (x[0] / L))**(n + 1)
        du = A * q * L * (h0 / dh) / (n + 1)
        return (u0 + du, 0.0)

    # Perturb the exact velocity for an initial guess.
    def velocity_guess(x):
        px, py = x[0] / L, x[1] / W
        q = 16 * px * (1 - px) * py * (1 - py)
        v = velocity_exact(x)
        return (v[0] + 60 * q * (px - 0.5), v[1])

    delta_x, error = [], []
    ice_shelf = icepack.models.IceShelf()
    opts = {"dirichlet_ids": [1, 3, 4], "tol": 1e-12}
    for N in range(16, 97, 4):
        mesh = firedrake.RectangleMesh(N, N, L, W)
        V = firedrake.VectorFunctionSpace(mesh, 'CG', 2)
        Q = firedrake.FunctionSpace(mesh, 'CG', 2)

        u_exact = icepack.interpolate(velocity_exact, V)
        u_guess = icepack.interpolate(velocity_guess, V)
        h = icepack.interpolate(lambda x: h0 - dh * x[0] / L, Q)
        A = icepack.interpolate(lambda x: icepack.rate_factor(T), Q)

        u = ice_shelf.diagnostic_solve(h=h, A=A, u0=u_guess, **opts)
        error.append(norm(u_exact - u) / norm(u_exact))
        delta_x.append(L / N)

        print(delta_x[-1], error[-1])

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print(slope, intercept)
    assert slope > 1.95
