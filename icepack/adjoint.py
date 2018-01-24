
import firedrake

def derivative(J, F, u, p, bc):
    """Compute the sensitivity of a functional of the solution of a PDE
    with respect to a parameter

    Given a functional :math:`J(u)` of the solution :math:`u` of the
    nonlinear equation

    ..math::
       F(u, p) = 0,

    this procedure returns a form representing the derivative of :math:`J`
    with respect to the parameter :math:`p` using the adjoint method.
    """
    dF = firedrake.derivative(F, u)
    dJ = firedrake.derivative(J, u)

    v = firedrake.Function(u.ufl_function_space())
    firedrake.solve(firedrake.adjoint(dF) == -dJ, v, bc)

    return firedrake.action(firedrake.adjoint(firedrake.derivative(F, p)), v)

