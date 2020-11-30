# Copyright (C) 2017-2020 by Daniel Shapero <shapero@uw.edu>
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

r"""Functions for calculating the viscosity of a glacier

This module contains procedures for computing the viscosity of a glacier
and, in particular, the viscous part of the action functional for ice flow.
Several flow models all have essentially the same viscous part.
"""

import warnings
import numpy as np
import firedrake
from firedrake import grad, sqrt, Identity, inner, sym, tr as trace
from icepack.constants import year, ideal_gas as R, glen_flow_law as n

transition_temperature = 263.15      # K
A0_cold = 3.985e-13 * year * 1.0e18  # mPa**-3 yr**-1
A0_warm = 1.916e3 * year * 1.0e18
Q_cold = 60                          # kJ / mol
Q_warm = 139


def rate_factor(T):
    r"""Compute the rate factor in Glen's flow law for a given temperature

    The strain rate :math:`\dot\varepsilon` of ice resulting from a stress
    :math:`\tau` is

    .. math::
       \dot\varepsilon = A(T)\tau^3

    where :math:`A(T)` is the temperature-dependent rate factor:

    .. math::
       A(T) = A_0\exp(-Q/RT)

    where :math:`R` is the ideal gas constant, :math:`Q` has units of
    energy per mole, and :math:`A_0` is a prefactor with units of
    pressure :math:`\text{MPa}^{-3}\times\text{yr}^{-1}`.

    Parameters
    ----------
    T : float, np.ndarray, or UFL expression
        The ice temperature

    Returns
    -------
    A : same type as T
        The ice fluidity
    """
    import ufl
    if isinstance(T, ufl.core.expr.Expr):
        cold = firedrake.lt(T, transition_temperature)
        A0 = firedrake.conditional(cold, A0_cold, A0_warm)
        Q = firedrake.conditional(cold, Q_cold, Q_warm)
        A = A0 * firedrake.exp(-Q / (R * T))
        if isinstance(T, firedrake.Constant):
            return firedrake.Constant(A)

        return A

    cold = T < transition_temperature
    warm = ~cold if isinstance(T, np.ndarray) else (not cold)
    A0 = A0_cold * cold + A0_warm * warm
    Q = Q_cold * cold + Q_warm * warm

    return A0 * np.exp(-Q / (R * T))


def membrane_stress(ε, A):
    r"""Calculate the membrane stress for a given strain rate and
    fluidity"""
    I = Identity(2)
    tr_ε = trace(ε)
    ε_e = sqrt((inner(ε, ε) + tr_ε**2) / 2)
    μ = 0.5 * A**(-1 / n) * ε_e**(1 / n - 1)
    return 2 * μ * (ε + tr_ε * I)


def viscosity_depth_averaged(u=None, h=None, A=None, **kwargs):
    r"""Return the viscous part of the action for depth-averaged models

    The viscous component of the action for depth-averaged ice flow is

    .. math::
        E(u) = \frac{n}{n+1}\int_\Omega h\cdot
        M(\dot\varepsilon, A):\dot\varepsilon\; dx

    where :math:`M(\dot\varepsilon, A)` is the membrane stress tensor

    .. math::
        M(\dot\varepsilon, A) = A^{-1/n}|\dot\varepsilon|^{1/n - 1}
        (\dot\varepsilon + \text{tr}\dot\varepsilon\cdot I).

    This form assumes that we're using the fluidity parameter instead
    the rheology parameter, the temperature, etc. To use a different
    variable, you can implement your own viscosity functional and pass it
    as an argument when initializing model objects to use your functional
    instead.

    Parameters
    ----------
    velocity : firedrake.Function
    thickness : firedrake.Function
    fluidity : firedrake.Function

    Returns
    -------
    firedrake.Form
    """
    # NOTE: This mess is for backwards-compatibility, so users can still pass
    # in the velocity, thickness, and fluidity as positional arguments if they
    # are still using old code.
    if (u is not None) or (h is not None) or (A is not None):
        warnings.warn("Abbreviated names (u, h, A) have been deprecated, use "
                      "full names (velocity, thickness, fluidity) instead.",
                      FutureWarning)

    if u is None:
        u = kwargs['velocity']
    if h is None:
        h = kwargs['thickness']
    if A is None:
        A = kwargs['fluidity']

    ε = sym(grad(u))
    M = membrane_stress(ε, A)
    return n / (n + 1) * h * inner(M, ε)
