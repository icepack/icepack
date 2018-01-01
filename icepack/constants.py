"""Physical constants

This module constains physical constants used throughout the library such
as the acceleration due to gravity, the universal gas constant, the density
of ice and water, etc.
Like Elmer/Ice, we use units of megapascals, meters, and years throughout.
This choice of units makes the rate factor
(see :py:func:`icepack.viscosity.rate_factor`) roughly equal to 1 for
typical temperatures.
"""

year = 365.25 * 24 * 60 * 60
gravity = 9.81 * year**2                # m/yr^2

#: density of ice
rho_ice = 917 / year**2 * 1.0e-6

#: density of seawater
rho_water = 1024 / year**2 * 1.0e-6

#: `ideal gas constant <https://en.wikipedia.org/wiki/Gas_constant>`_
ideal_gas = 8.3144621e-3                # kJ / mol K

#: exponent in the nonlinear constitutive law for ice
glen_flow_law = 3.0

