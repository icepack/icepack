
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

### Contents

* `field_type.hpp`: definition of the `FieldType` class template
* `field_interpolate.hpp`: procedures for interpolating deal.II `Function` or `TensorFunction` objects into finite element fields
* `field_algebra.hpp`: procedures for performing algebraic manipulations on fields, e.g. addition, subtraction, multiplication by scalars, etc.
* the header `icepack/field.hpp` aggregates all of the headers in this directory for easy including
