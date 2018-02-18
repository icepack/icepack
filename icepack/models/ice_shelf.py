
import firedrake
from firedrake import div, dx
from icepack.constants import rho_ice, rho_water, gravity as g
from icepack.models.viscosity import viscosity_depth_averaged as viscosity
from icepack.models.mass_transport import MassTransport
from icepack.optimization import newton_search
from icepack.utilities import add_kwarg_wrapper


def gravity(u=None, h=None):
    """Return the gravitational part of the ice shelf action functional

    The gravitational part of the ice shelf action functional is

    .. math::
        E(u) = \\frac{1}{2}\int_\Omega \\varrho gh^2\\nabla\cdot u\hspace{2pt}dx

    Parameters
    ----------
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

    This class provides functions that solve for the velocity and
    thickness of a floating ice shelf. The relevant physics can be found
    in ch. 6 of Greve and Blatter.

    .. seealso::
       :py:func:`icepack.models.viscosity.viscosity_depth_averaged`
          Default implementation of the ice shelf viscous action
    """
    def __init__(self, viscosity=viscosity, gravity=gravity):
        self.mass_transport = MassTransport()
        self.viscosity = add_kwarg_wrapper(viscosity)
        self.gravity = add_kwarg_wrapper(gravity)

    def action(self, u=None, h=None, **kwargs):
        """Return the action functional that gives the ice shelf diagnostic
        model as the Euler-Lagrange equations

        The action functional for the ice shelf diagnostic model is

        .. math::
            E(u) = \int_\Omega\left(\\frac{n}{n + 1}hM:\dot\\varepsilon
            - \\frac{1}{2}\\varrho gh^2\\nabla\cdot u\\right)dx

        where :math:`u` is the velocity, :math:`h` is the ice thickness,
        :math:`\dot\\varepsilon` is the strain-rate tensor, and :math:`M` is
        the membrane stress tensor.

        Parameters
        ----------
        u : firedrake.Function
            ice velocity
        h : firedrake.Function
            ice thickness

        Returns
        -------
        E : firedrake.Form
            the ice shelf action functional

        Other parameters
        ----------------
        **kwargs
            All other keyword arguments will be passed on to the viscosity
            and gravity functionals. The ice fluidity coefficient, for
            example, is passed as a keyword argument.
        """
        viscosity = self.viscosity(u=u, h=h, **kwargs)
        gravity = self.gravity(u=u, h=h, **kwargs)
        return viscosity - gravity

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
            `viscosity` and `gravity` functions that were set when this
            model object was initialized
        """
        u = u0.copy(deepcopy=True)

        # Scale the non-dimensional convergence tolerance by the viscous power
        viscosity = self.viscosity(u=u, h=h, **kwargs)
        scale = firedrake.assemble(viscosity)
        tolerance = tol * scale

        # Create boundary conditions
        bcs = [firedrake.DirichletBC(u.function_space(), (0, 0), k)
               for k in dirichlet_ids]

        # Solve the nonlinear optimization problem
        return newton_search(self.action(u=u, h=h, **kwargs), u, bcs, tolerance)

    def prognostic_solve(self, dt, h0=None, a=None, u=None, **kwargs):
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
        return self.mass_transport.solve(dt, h0=h0, a=a, u=u, **kwargs)

    def suggested_elements(self, degree, cell=firedrake.triangle):
        """Return a dictionary of suggested finite elements for each field

        The suggested finite elements will work for this particular model, and
        you can experiment as you see fit. For simple models like ice shelves,
        any continuous element will work fine. For the full Stokes system, on
        the other hand, many velocity/pressure element pairs are unstable.
        """
        return {"thickness": firedrake.FiniteElement('CG', cell, degree),
                "velocity": firedrake.VectorElement('CG', cell, degree)}

