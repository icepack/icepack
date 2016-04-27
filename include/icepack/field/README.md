
# Fields

This directory contains class templates for representing spatial fields, which represent things like the temperature distribution, thickness or velocity of an ice sheet.
These are represented by the class template `FieldType`, which can describe physical quantities in any dimension or of any tensor rank.
For example, scalar fields like temperature or thickness have rank 0, while vector fields like the ice velocity have rank 1.

Rather than have to type the unintuitive name `FieldType` every time, we also provide the aliases

    Field<dim> = FieldType<0, dim>
    VectorField<dim> = FieldType<1, dim>

which are the forms encountered most throughout the rest of the library.
We only use the general form `FieldType` for operations (e.g. interpolation or algebraic manipulation) which can be defined in the same way independent of the underlying rank.

Every field is defined with respect to some `Discretization` object, which aggregates various sundry data such as a degree-of-freedom handler, finite element, etc.
See the header `discretization.hpp`.

In addition to the spatial dimension and tensor rank, `FieldType` has one additional template argument, an enum `Duality` which can take one of the values `primal` or `dual`.
Primal fields are the ones you are most familiar with; you can evaluate them at a point.
Since they are the most common case, the duality argument defaults to `primal`.
Dual fields often arise as the product of a differential operator, such as the Laplacian, and a primal field.
This distinction is enforced in order to avoid hard-to-detect errors that can occur by forgetting to multiply by either the mass matrix or its inverse.
As for primal fields, we provide the type aliases

    DualField<dim> = FieldType<0, dim, dual>
    DualVectorField<dim> = FieldType<1, dim, dual>

for succinctness.
For most usage of icepack, you will probably never have to use a `DualField` or a `DualVectorField`.
See the full documentation for `FieldType` for more explanation of the underlying mathematics.


### Contents

* `field_type.hpp`: definition of the `FieldType` class template
* `field_interpolate.hpp`: procedures for interpolating deal.II `Function` or `TensorFunction` objects into finite element fields
* `field_algebra.hpp`: procedures for performing algebraic manipulations on fields, e.g. addition, subtraction, multiplication by scalars, etc.
* the header `icepack/field.hpp` aggregates all of the headers in this directory for easy including
