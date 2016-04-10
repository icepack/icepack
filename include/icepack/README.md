
# Icepack class overview

This directory contains all the header files and class definitions for `icepack`.
The most important classes for the typical user are:

* `Field, VectorField`: classes for representing things like the ice thickness, temperature, velocity, etc. See `field.hpp` and the headers under `field/`.
* `IceShelf`, `IceStream`, `DepthAveragedModel`: classes for computing the ice thickness and velocity from well-known models of glacier flow. See the headers in `glacier_models/`.

These are in turn built on top of assorted lower-level functions:

* viscosity and basal shear parameterizations are in the directory `physics/`
* a class for representing gridded data and functions for reading geotif and Arc ASCII files are in `grid_data.hpp`
* the `Discretization` class template in the header `discretization.hpp` is a support class for discretizing finite element fields
