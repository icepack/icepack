
from numpy import exp
from icepack.constants import year, ideal_gas as R

transition_temperature = 263.15     # K
A0_cold = 3.985e-13 * year * 1.0e18 # mPa**-3 yr**-1
A0_warm = 1.916e3 * year * 1.0e18
Q_cold = 60                         # kJ / mol
Q_warm = 139

def rate_factor(T):
    """Compute the rate factor in Glen's flow law for a given temperature

    The strain rate :math:`\dot\\varepsilon` of ice resulting from a stress
    :math:`\\tau` is

    .. math::

       \dot\\varepsilon = A(T)\\tau^3

    where :math:`A(T)` is the temperature-dependent rate factor:

    .. math::

       A(T) = A_0\exp(-Q/RT)

    where :math:`R` is the ideal gas constant, :math:`Q` has units of
    energy per mole, and :math:`A_0` is a prefactor with units of
    pressure :math:`\\text{MPa}^{-3}\\times\\text{yr}^{-1}`.
    """
    cold = T < transition_temperature
    A0 = A0_cold if cold else A0_warm
    Q = Q_cold if cold else Q_warm
    return A0 * exp(-Q / (R * T))

