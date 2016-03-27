
# Physics

This directory contains data and class templates for various constituent physical processes of glacier models.
For example, the `BasalShear` class defined in the header `basal_shear.hpp` defines the parameterization of stress at a glacier bed as a function of the sliding rheology and yield stress/speed.
Likewise for the classes `Rheology` and `ConstitutiveTensor`.

The various physics parameterizations can also be linearized about some background state so that we can use e.g. Newton's method.
For example, the viscosity is a nonlinear function of the strain rate (this is Glen's flow law), but we need to linearize it about a current guess of the ice velocity in order to solve the diagnostic equations via Newton's method.
The gradient of the objective functional in various data assimilation problems is also a function of the linearized physics.
Whether to evaluate the full physics or its linearized form is determined by a template parameter, the enum `Linearity`.


### Contents

* `constants.hpp`: various physical constants, such as the density of ice and water, the gas constant, etc.
These are all in units of megapascals / meters / seconds, because that happens to make a lot of things work out to nice round numbers.
* `linearity.hpp`: defines an enum variable for whether the physics are or are not linearized
* `viscosity.hpp`: classes for the ice rheology and the constitutive tensor, which relates the ice strain rate and stress; used in all glacier models
* `basal_shear.hpp`: parameterization of the stress due to basal sliding; used in the ice stream glacier model, but not for ice shelves
