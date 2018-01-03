
import numpy as np
import firedrake
from firedrake import Expression, Function, interpolate as _interpolate

def interpolate(f, Q):
    """Interpolate an analytically-defined function to a function space

    Parameters
    ----------
    f : firedrake.Expression, firedrake.Function, or a callable object
        If callable, must take in values of the dimension of the function
        space and return either scalar/vector/tensor values
    Q : firedrake.FunctionSpace
        The function space to interpolate to

    The function space `Q` must be of the right rank for the type of the
    range of `f`. For example, if `f` is a vector-valued function, then `Q`
    must be a vector function space, likewise for tensors or scalars.

    Returns
    -------
    firedrake.Function
        A finite element function defined on `Q` with the same nodal values
        as the function `f`
    """
    if isinstance(f, (firedrake.Expression, firedrake.Function)):
        return firedrake.interpolate(f, Q)

    if hasattr(f, '__call__'):
        domain = Q.ufl_domain()
        element = Q.ufl_element()
        if len(element.sub_elements()) > 0:
            element = element.sub_elements()[0]

        V = firedrake.VectorFunctionSpace(domain, element)
        X = firedrake.interpolate(domain.coordinates, V)

        q = firedrake.Function(Q)

        # TODO: make this work for vectorized things & get rid of the loop
        n = q.dat.data.shape[0]
        for i in range(n):
            q.dat.data[i] = f(X.dat.data_ro[i, :])

        return q

    raise ValueError('Argument must be callable!')

