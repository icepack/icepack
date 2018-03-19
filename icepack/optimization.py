# Copyright (C) 2017-2018 by Daniel Shapero <shapero@uw.edu>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

from firedrake import action, assemble, replace, derivative, solve, \
    Function, Constant

def compute_search_direction(E, u, bc):
    """Solve the linear system for the search direction in Newton's method

    Parameters
    ----------
    E : firedrake.Form
        A functional to be optimized
    u : firedrake.Function
        The current guess for the optimizer of `E`
    bc : firedrake.DirichletBC
        Boundary conditions for the PDE

    Returns
    -------
    firedrake.Function
        The solution of the PDE

    .. math:: (d^2E)p = -dE
    """
    F = derivative(E, u)
    dF = derivative(F, u)

    p = Function(u.function_space())
    solve(dF == -F, p, bc,
          solver_parameters={'ksp_type': 'cg', 'pc_type': 'ilu'})
    return p


def functional_along_line(E, u, p):
    """Restrict a functional to a line in parameter space

    Parameters
    ----------
    E : firedrake.Form
        The functional to be optimized
    u : firedrake.Function
        The current guess for the optimizer of `E`
    p : firedrake.Function
        The current search direction

    Returns
    -------
    callable
        The function `f(z) = E(u0 + z * p)`
    """
    alpha = Constant(0.0)
    F = replace(E, {u: u + alpha * p})
    def f(beta):
        alpha.assign(beta)
        return assemble(F)

    return f


def backtracking_search(E, u, p, armijo=1.0e-4, rho=0.5):
    """Search for the minimum of a functional along a line

    Parameters
    ----------
    E : firedrake.Form
        The functional to be minimized
    u : firedrake.Function
        The current guess for the minimizer of `E`
    p : firedrake.Function
        The direction from `u` where we are searching for a new minimizer
    armijo : float, optional
        The Armijo parameter used to determine when the line search has
        found a sufficiently better guess
    rho : float, optional
        The backtracking parameter used to determine how much to shrink the
        step length if the current value is too long

    Returns
    -------
    float
        The distance along `p` of an approximate minimizer
    """
    f = functional_along_line(E, u, p)
    f0, df0 = f(0.0), assemble(action(derivative(E, u), p))
    assert df0 < 0

    alpha = 1.0
    while f(alpha) > f0 + armijo * alpha * df0:
        alpha *= rho

    return alpha


def newton_search(E, u, bc, tolerance, max_iterations=50):
    """Find the minimizer of a convex functional

    Parameters
    ----------
    E : firedrake.Form
        The functional to be minimized
    u0 : firedrake.Function
        Initial guess for the minimizer
    tolerance : float
        Stopping criterion for the optimization procedure
    max_iterations : int, optional
        Optimization procedure will stop at this many iterations regardless
        of convergence

    Returns
    -------
    firedrake.Function
        The approximate minimizer of `E` to within tolerance
    """
    dE = derivative(E, u)

    n = 0
    while True:
        p = compute_search_direction(E, u, bc)
        dE_dp = abs(assemble(action(dE, p)))

        if (dE_dp < tolerance) or (n >= max_iterations):
            return u

        alpha = backtracking_search(E, u, p)
        u.assign(u + alpha * p)
        n += 1

