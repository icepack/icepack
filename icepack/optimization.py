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

import firedrake


def newton_search(E, u, bc, tolerance, scale,
                  max_iterations=50, armijo=1e-4, contraction_factor=0.5,
                  form_compiler_parameters={},
                  solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'}):
    """Find the minimizer of a convex functional

    Parameters
    ----------
    E : firedrake.Form
        The functional to be minimized
    u0 : firedrake.Function
        Initial guess for the minimizer
    tolerance : float
        Stopping criterion for the optimization procedure
    scale : firedrake.Form
        A positive scale functional by which to measure the objective
    max_iterations : int, optional
        Optimization procedure will stop at this many iterations regardless
        of convergence
    armijo : float, optional
        The constant in the Armijo condition (see Nocedal and Wright)
    contraction_factor : float, optional
        The amount by which to backtrack in the line search if the Armijo
        condition is not satisfied
    form_compiler_parameters : dict, optional
        Extra options to pass to the firedrake form compiler
    solver_parameters : dict, optional
        Extra options to pass to the linear solver

    Returns
    -------
    firedrake.Function
        The approximate minimizer of `E` to within tolerance
    """
    F = firedrake.derivative(E, u)
    H = firedrake.derivative(F, u)
    p = firedrake.Function(u.function_space())
    dE_dp = firedrake.action(F, p)

    def assemble(*args, **kwargs):
        return firedrake.assemble(
            *args, **kwargs, form_compiler_parameters=form_compiler_parameters)

    n = 0
    while True:
        # Compute a search direction
        firedrake.solve(H == -F, p, bc,
                        solver_parameters=solver_parameters,
                        form_compiler_parameters=form_compiler_parameters)

        # Compute the directional derivative, check if we're done
        slope = assemble(dE_dp)
        assert slope < 0
        if (abs(slope) < assemble(scale) * tolerance) or (n >= max_iterations):
            return u

        # Backtracking search
        E0 = assemble(E)
        α = firedrake.Constant(1)
        Eα = firedrake.replace(E, {u: u + α * p})
        while assemble(Eα) > E0 + armijo * α.values()[0] * slope:
            α.assign(α * contraction_factor)

        u.assign(u + α * p)
        n += 1

