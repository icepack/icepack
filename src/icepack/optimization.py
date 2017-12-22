
from firedrake import (
    action, assemble, replace, derivative,
    Function, Constant, solve
)

def compute_search_direction(E, u, bc):
    p = Function(u.function_space())
    F = derivative(E, u)
    dF = derivative(F, u)
    solve(dF == -F, p, bc)
    return p

def functional_along_line(E, u, p):
    alpha = Constant(0.0)
    F = replace(J, {u: u + alpha * p})
    def f(beta):
        alpha.assign(beta)
        return assemble(F)

    return f

def backtracking_search(E, u, p, armijo=1.0e-4, rho=0.5):
    f = functional_along_line(E, u, p)
    f0, df0 = f(0.0), assemble(action(derivative(E, u), p))

    alpha = 1.0
    while f(alpha) > f0 + armijo * alpha * df0:
        alpha *= rho

    return alpha

def newton_search(E, u, bc, tolerance, max_iterations=50):
    dE = derivative(E, u)

    n = 0
    while True:
        p = compute_search_direction(E, u, bc)
        dE_dp = math.fabs(assemble(action(dE, p)))

        if (dE_dp < tolerance) or (n >= max_iterations):
            return u

        alpha = backtracking_search(E, u, p)
        u.assign(u + alpha * p)
        n += 1

