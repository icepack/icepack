
# Glacier models

This directory contains classes for large-scale flow modeling of glaciers and ice sheets.
The two most important things that any glacier model has to do are:

* given the glacier thickness and inflow velocities, compute the velocity throughout the glacier
* given the current glacier thickness, velocity and accumulation rate, compute the ice thickness at a later time

Consequently, every glacier model has a function

    VectorField u = model.diagnostic_solve(h, ...)
    
to compute the ice velocity as a function of thickness and possibly other parameters; and another function

    Field h = model.prognostic_solve(dt, h0, a, u, ...)

to update the old ice thickness `h0` by a timestep `dt.`
See the header files `ice_shelf.hpp` and `ice_stream.hpp` for which fields are needed in each case, as there are slight variations among models.
For example, only for a grounded ice stream does the user need to supply a basal friction parameter when computing the current velocity, while in both cases one must input a temperature field.
Despite these minor discrepancies, the diagnostic and prognostic solve procedures are the most important functions in `icepack`.

The `DepthAveragedModel` class contains some common functions that are used by any depth-averaged glacier model, such as propagating the ice thickness forward in time according to the prognostic equation.
Solving the PDE for the ice velocity is left to child classes of `DepthAveragedModel`.

The `IceShelf` class is appropriate for modeling floating ice shelves.
The `IceStream` class can simulate floating ice shelves, but can additionally model fast-sliding grounded ice streams.
Both of these are depth-averaged models.

All of the glacier models used herein rely on physics parameterizations in the directory `icepack/physics/`.
