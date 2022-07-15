# Copyright (C) 2017-2019 by Daniel Shapero <shapero@uw.edu>
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

#: number of seconds in a year, for unit conversions
year = 365.25 * 24 * 60 * 60

#: acceleration due to gravity (m / yr^2)
gravity = 9.81 * year**2

#: density of ice
ice_density = 917 / year**2 * 1.0e-6

#: density of seawater
water_density = 1024 / year**2 * 1.0e-6

#: ideal gas constant (kJ / mol K)
ideal_gas = 8.3144621e-3

#: exponent in the nonlinear constitutive law for ice
glen_flow_law = 3.0

#: regularizing strain rate in Glen law (1 / yr)
strain_rate_min = 1e-5

#: exponent in the nonlinear friction law for ice sliding
weertman_sliding_law = 3.0

#: specific heat capacity of ice at -10C (m^2 / yr^2 / K)
heat_capacity = 2.0e3 * year**2

#: thermal diffusivity of ice at -10C (m^2 / yr)
thermal_diffusivity = 2.3e-3 / (917 * 2.0) * year

#: melting point of ice at atmospheric pressure (K)
melting_temperature = 273.15

#: latent heat of melting of ice (m^2 / yr^2)
latent_heat = 334e3 * year**2
