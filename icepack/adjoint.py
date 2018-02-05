
import firedrake

def derivative(J, F, u, p, bc=None):
    """Compute the sensitivity of a functional of the solution of a PDE
    with respect to a parameter

    Let :math:`G(p)` be the solution operator for the nonlinear equation

    .. math:: F(u, p) = 0,

    i.e. :math:`F(G(p), p) = 0` for all :math:`p`. Given a functional
    :math:`J(u)` of the solution, this function returns the derivative of
    :math:`J(G(p))` with respect to the parameter :math:`p` using the
    adjoint method.

    Parameters
    ----------
    J : firedrake.Form
        A functional of the solution of a PDE
    F : firedrake.Form
        The weak form of the PDE
    u : firedrake.Function
        A solution of the PDE with the given parameters
    p : firedrake.Function
        The parameters we're linearizing around
    bc : firedrake.DirichletBC, optional
        Boundary conditions for the PDE

    Returns
    -------
    dJ : firedrake.Form
        The derivative of `J` with respect to `p`

    """
    dF = firedrake.derivative(F, u)
    dJ = firedrake.derivative(J, u)

    v = firedrake.Function(u.ufl_function_space())
    firedrake.solve(firedrake.adjoint(dF) == -dJ, v, bc)

    return firedrake.action(firedrake.adjoint(firedrake.derivative(F, p)), v)

