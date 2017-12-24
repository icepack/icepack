
from numpy import exp
from icepack.constants import year, ideal_gas as R

transition_temperature = 263.15     # K
A0_cold = 3.985e-13 * year * 1.0e18 # mPa**-3 yr**-1
A0_warm = 1.916e3 * year * 1.0e18
Q_cold = 60                         # kJ / mol
Q_warm = 139

def rate_factor(T):
    cold = T < transition_temperature
    A0 = A0_cold if cold else A0_warm
    Q = Q_cold if cold else Q_warm
    return A0 * exp(-Q / (R * T))

