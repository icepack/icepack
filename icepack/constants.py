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

r"""Physical constants

This module constains physical constants used throughout the library such
as the acceleration due to gravity, the universal gas constant, the density
of ice and water, etc.
Like Elmer/Ice, we use units of megapascals, meters, and years throughout.
This choice of units makes the rate factor
(see :py:func:`icepack.models.viscosity.rate_factor`) roughly equal to 1 for
typical temperatures.
"""

year = 365.25 * 24 * 60 * 60
gravity = 9.81 * year**2                # m/yr^2

#: density of ice
rho_ice = 917 / year**2 * 1.0e-6

#: density of seawater
rho_water = 1024 / year**2 * 1.0e-6

#: ideal gas constant (kJ / mol K)
ideal_gas = 8.3144621e-3

#: exponent in the nonlinear constitutive law for ice
glen_flow_law = 3.0

#: exponent in the nonlinear friction law for ice sliding
weertman_sliding_law = 3.0

