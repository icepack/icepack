
import numpy as np

# Test that our supposed analytical solution really is a solution using the
# symbolic algebra package `sympy`.
def test_manufactured_solution():
    from sympy import symbols, simplify, diff, lambdify
    from icepack import constants
    L, rho_I, rho_W = symbols('L rho_I rho_W', real=True, positive=True)
    B, g = symbols('B g', real=True, positive=True)
    n, m = symbols('n m', integer=True, positive=True)

    def membrane_stress(x, u, B):
        return simplify(2 * B * diff(u, x)**(1/n))

    def friction(x, u, C):
        return simplify(-C * u**(1/m))

    def driving_stress(x, h, s):
        return simplify(-rho_I * g * h * diff(s, x))

    def shelfy_stream_eqns(x, u, h, s, B, C):
        return simplify(diff(h * membrane_stress(x, u, B), x) +
                        friction(x, u, C) +
                        driving_stress(x, h, s))

    def boundary_condition(x, u, h, s, B):
        M = membrane_stress(x, u, B)
        d = (s - h).subs(x, L)
        tau = (rho_I * g * h**2 - rho_W * g * d**2) / 2
        return simplify((h * M - tau).subs(x, L))

    x = symbols('x', real=True)

    h0, dh = symbols('h0 dh', real=True, positive=True)
    h = h0 - dh * x / L

    hf = symbols('hf', real=True, positive=True)
    d = -rho_I / rho_W * h.subs(x, L) + hf
    rho = (rho_I - rho_W * d**2 / h**2).subs(x, L)

    u0 = symbols('u0', real=True, positive=True)
    du = (rho*g*h0/(4*B))**n * (1 - (1 - dh/h0*x/L)**(n+1)) * L * (h0/dh)/(n+1)
    u = u0 + du

    beta = 1/2
    alpha = beta * rho / rho_I * dh / L
    C = alpha * (rho_I * g * h) * u**(-1/m)

    ds = (1 + beta) * rho / rho_I * dh
    s = d + h.subs(x, L) + ds * (1 - x / L)

    T = 254.15
    rheology = icepack.rate_factor(T)**(-1/constants.glen_flow_law)
    values = {u0: 100, dh: 100, h0: 500, L: 20e3, hf: 10, B: rheology,
              rho_I: constants.rho_ice, rho_W: constants.rho_water,
              n: constants.glen_flow_law, m: constants.weertman_sliding_law,
              g: constants.gravity}


    tau_b = lambdify(x, friction(x, u, C).subs(values), "numpy")
    tau_d = lambdify(x, driving_stress(x, h, s).subs(values), "numpy")
    M = membrane_stress(x, u, B)
    tau_m = lambdify(x, simplify(diff(h * M, x)).subs(values), "numpy")
    xs = np.linspace(0, values[L], 21)

    tolerance = 1e-8
    assert abs(boundary_condition(x, u, h, s, B).subs(values)) < tolerance
    assert (np.max(np.abs(tau_m(xs) + tau_b(xs) + tau_d(xs)))
            < tolerance * np.max(np.abs(tau_d(xs))))


# Now test our numerical solvers against this analytical solution.
import firedrake
import icepack, icepack.models
from icepack.constants import gravity, rho_ice, rho_water, \
    glen_flow_law as n, weertman_sliding_law as m

def test_diagnostic_solver_convergence():
    L, W = 20.0e3, 20.0e3
    h0, dh = 500.0, 100.0
    def thickness(x):
        return h0 - dh * x[0] / L

    # Pick a height above flotation at the ice terminus. In order to have an
    # exact ice velocity of the same form as the exact solution for an ice
    # shelf, we have to pick the pseudo-density to be a certain value for the
    # velocity to satisfy the boundary condition at the terminus.
    height_above_flotation = 10.0
    d = -rho_ice / rho_water * thickness((L, W/2)) + height_above_flotation
    rho = rho_ice - rho_water * d**2 / thickness((L, W/2))**2

    # We'll arbitrarily pick this to be the velocity, then we'll find a
    # friction coefficient and surface elevation that makes this velocity
    # an exact solution of the shelfy stream equations.
    T = 254.15
    u0 = 100.0
    def velocity_exact(x):
        A = icepack.rate_factor(T) * (rho * gravity * h0 / 4)**n
        q = 1 - (1 - (dh/h0) * (x[0]/L))**(n + 1)
        du = A * q * L * (h0/dh) / (n + 1)
        return (u0 + du, 0.0)

    def velocity_guess(x):
        px, py = x[0] / L, x[1] / W
        q = 16 * px * (1 - px) * py * (1 - py)
        v = velocity_exact(x)
        return (v[0] + 60 * q * (px - 0.5), v[1])

    # With this choice of friction coefficient, we can take the surface
    # elevation to be a linear function of the horizontal coordinate and the
    # velocity will be an exact solution of the shelfy stream equations.
    beta = 1/2
    alpha = beta * rho / rho_ice * dh / L
    def friction(x):
        u = velocity_exact(x)[0]
        h = thickness(x)
        return alpha * (rho_ice * gravity * h) * u**(-1/m)

    ds = (1 + beta) * rho/rho_ice * dh
    def surface(x):
        return d + h0 - dh + ds * (1 - x[0] / L)

    # Create an ice stream model.
    ice_stream = icepack.models.IceStream()
    opts = {'dirichlet_ids': [1, 3, 4], 'tol': 1e-12}

    # Solve the ice stream model for successively higher mesh resolution.
    delta_x, error = [], []
    norm = lambda v: icepack.norm(v, norm_type='H1')

    for N in range(16, 65, 4):
        mesh = firedrake.RectangleMesh(N, N, L, W)
        degree = 2
        Q = firedrake.FunctionSpace(mesh, 'CG', degree)
        V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)

        u_exact = icepack.interpolate(velocity_exact, V)
        u_guess = icepack.interpolate(velocity_guess, V)
        h = icepack.interpolate(thickness, Q)
        s = icepack.interpolate(surface, Q)
        C = icepack.interpolate(friction, Q)
        A = icepack.interpolate(lambda x: icepack.rate_factor(T), Q)

        u = ice_stream.diagnostic_solve(h=h, s=s, A=A, C=C, u0=u_guess, **opts)
        error.append(norm(u_exact - u) / norm(u_exact))
        delta_x.append(L / N)

        print(delta_x[-1], error[-1])

    log_delta_x = np.log2(np.array(delta_x))
    log_error = np.log2(np.array(error))
    slope, intercept = np.polyfit(log_delta_x, log_error, 1)

    print(slope, intercept)
    assert slope > degree - 0.05


def test_computing_surface():
    L, W = 20.0e3, 20.0e3
    h0, dh = 500.0, 100.0
    def thickness(x):
        return h0 - dh * x[0] / L

    b0 = rho_ice / rho_water * (dh / 2 - h0)

    N = 16
    mesh = firedrake.RectangleMesh(N, N, L, W)

    degree = 2
    Q = firedrake.FunctionSpace(mesh, 'CG', degree)

    h = icepack.interpolate(thickness, Q)
    b = icepack.interpolate(lambda x: b0, Q)

    ice_stream = icepack.models.IceStream()
    s = ice_stream.compute_surface(h=h, b=b)
    x = (L/2, W/2)
    assert abs(s(x) - (1 - rho_ice/rho_water) * h(x)) < 1e-8

