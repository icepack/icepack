
import numpy as np
from firedrake import Expression, Function, interpolate as _interpolate

def _func_wrapper_type(rank, dim):
    class _FunctionWrapper(Expression):
        def __init__(self, q, *args, **kwargs):
            Expression.__init__(self, *args, **kwargs)
            self._q = q

        def eval(self, value, x):
            value[:] = np.reshape(self._q(x), dim ** rank)

        def value_shape(self):
            return tuple((dim for i in range(rank)))

    return _FunctionWrapper


def interpolate(q, X):
    """Interpolate an analytically-defined function to a function space

    Parameters
    ----------
    q
        Either a `firedrake.Expression` or a callable object, taking in an
        argument `x` and returning a scalar, vector, or tensor
    X : firedrake.FunctionSpace
        The function space that `q` will be interpolated to. If `q` is a
        scalar/vector/tensor field, `X` must be a function space of the
        right rank.

    Returns
    -------
    firedrake.Function
        A finite element function defined on `X` with the same nodal values
        as the function `q`
    """
    if isinstance(q, Expression) or isinstance(q, Function):
        return _interpolate(q, X)

    if hasattr(q, '__call__'):
        domain = X.ufl_domain()
        dim = domain.topological_dimension()
        if dim != domain.geometric_dimension():
            raise ValueError('Geometric and topological dimension of function'
                             'spaces must be equal!')

        element = X.ufl_element()
        rank = len(element.value_shape())

        f = _func_wrapper_type(rank, dim)(q)
        return _interpolate(f, X)

    raise ValueError('Argument must be callable!')

