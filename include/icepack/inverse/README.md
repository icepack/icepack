
# Inverse methods

This directory contains functions for inferring the temperature or basal shear stress of a glacier using inverse methods.
In particular, the `adjoint` method of the various glacier models is used to quickly compute the functional derivative of the model/data misfit.
The derivative is then used to optimize the inferred field, using e.g. gradient descent, conjugate gradient, etc.
