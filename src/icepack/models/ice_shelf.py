
import firedrake
from firedrake import grad, div, dx, sqrt, inner, sym, tr as trace
from icepack.constants import rho_ice, rho_water, \
    gravity as g, glen_flow_law as n
#from icepack.mass_transport import MassTransport
from icepack.optimization import newton_search


def M(eps, A):
    """Calculate the membrane stress for a given strain rate and fluidity"""
    I = firedrake.Identity(2)
    tr = trace(eps)
    eps_e = sqrt((inner(eps, eps) + tr * tr) / 2)
    mu = 0.5 * A**(-1/n) * eps_e**(1/n - 1)
    return 2 * mu * (eps + tr * I)


def eps(u):
    """Calculate the strain rate for a given flow velocity"""
    return sym(grad(u))


def viscosity(u=None, h=None, A=None, **kwargs):
    """Return the viscous part of the ice shelf action functional

    The viscous component of the action for ice shelf flow is

    .. math::
        E(u) = \\frac{n}{n+1}\int_\Omega h\cdot M(\dot\\varepsilon, A):\dot\\varepsilon\hspace{2pt} dx

    where :math:`M(\dot\\varepsilon, A)` is the membrane stress tensor

    .. math::
        M(\dot\\varepsilon, A) = A^{-1/n}|\dot\\varepsilon|^{1/n - 1}(\dot\\varepsilon + \\text{tr}\dot\\varepsilon\cdot I).

    This form assumes that we're using the fluidity parameter instead
    the rheology parameter, the temperature, etc. To use a different
    variable, you can subclass `IceShelf` and override this method.

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness
    A : firedrake.Function
        ice fluidity parameter

    Returns
    -------
    firedrake.Form
    """
    return n/(n + 1) * h * inner(M(eps(u), A), eps(u)) * dx


def gravity(u=None, h=None, **kwargs):
    """Return the gravitational part of the ice shelf action functional

    The gravitational part of the ice shelf action functional is

    .. math::
        E(u) = \\frac{1}{2}\int_\Omega \\varrho gh^2\\nabla\cdot u\hspace{2pt}dx

    Paramters
    ---------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness

    Returns
    -------
    firedrake.Form
    """
    rho = rho_ice * (1 - rho_ice / rho_water)
    return 0.5 * rho * g * h**2 * div(u) * dx


class IceShelf(object):
    """Class for modelling the flow of floating ice shelves

    This class provides functions that solve for the ice velocity and thickness.
    The relevant physics can be found in Greve and Blatter.
    """
    def __init__(self, viscosity=viscosity, gravity=gravity):
        #self.mass_transport = MassTransport()
        self.viscosity = viscosity
        self.gravity = gravity


    def diagnostic_solve(self, u0=None, h=None,
                         dirichlet_ids=[], tol=1e-6, **kwargs):
        """Solve for the ice velocity from the thickness

        Parameters
        ----------
        u0 : firedrake.Function
            Initial guess for the ice velocity; the Dirichlet boundaries
            are taken from `u0`
        h : firedrake.Function
            Ice thickness
        dirichlet_ids : list of int
            list of integer IDs denoting the parts of the boundary where
            Dirichlet conditions should be applied
        tol : float
            dimensionless tolerance for when to terminate Newton's method

        Returns
        -------
        u : firedrake.Function
            Ice velocity

        Other parameters
        ----------------
        **kwargs
            All other keyword arguments will be passed on to the
            `viscosity_functional` and `gravity_functional` methods used to
            define the action.
        """
        # Create the action functional for ice shelf flow
        u = u0.copy(deepcopy=True)
        viscosity = self.viscosity(u=u, h=h, **kwargs)
        gravity = self.gravity(u=u, h=h, **kwargs)

        # Scale the (non-dimensional) convergence tolerance by the viscous power
        scale = firedrake.assemble(viscosity)
        tolerance = tol * scale

        # Create boundary conditions
        bcs = [firedrake.DirichletBC(u.function_space(), (0, 0), k)
               for k in dirichlet_ids]

        # Solve the nonlinear optimization problem
        return newton_search(viscosity - gravity, u, bcs, tolerance)

    def prognostic_solve(self, dt, h0, a, u, **kwargs):
        """Propagate the ice thickness forward one timestep

        Parameters
        ----------
        dt : float
            The timestep length
        h0, a : firedrake.Function
            The initial ice thickness and the accumulation rate
        u : firedrake.Function
            The ice velocity

        Returns
        -------
        h : firedrake.Function
            The new ice thickness at `t + dt`
        """
        #return self.mass_transport.solve(dt, h0, a, u, **kwargs)

    def suggested_elements(self, degree, cell=firedrake.triangle):
        """Return a dictionary of suggested finite elements for each field

        The suggested finite elements will work for this particular model, and
        you can experiment as you see fit. For simple models like ice shelves,
        any continuous element will work fine. For the full Stokes system, on
        the other hand, many velocity/pressure element pairs are unstable.
        """
        return {"thickness": firedrake.FiniteElement('CG', cell, degree),
                "velocity": firedrake.VectorElement('CG', cell, degree)}

