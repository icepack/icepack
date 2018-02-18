
import firedrake
from firedrake import inner, grad, div, dx, ds, sqrt
from icepack.constants import rho_ice, rho_water, \
    gravity as g, weertman_sliding_law as m
from icepack.models.viscosity import viscosity_depth_averaged as viscosity
from icepack.models.mass_transport import MassTransport
from icepack.optimization import newton_search
from icepack.utilities import add_kwarg_wrapper


def tau_b(u, C):
    """Compute the basal shear stress for a given sliding velocity
    """
    return -C * sqrt(inner(u, u))**(1/m - 1) * u


def friction(u=None, C=None):
    """Return the frictional part of the ice stream action functional

    The frictional part of the ice stream action functional is

    .. math::
       E(u) = -\\frac{m}{m + 1}\int_\Omega\\tau_b(u, C)\cdot u\hspace{2pt}dx

    where :math:`\\tau_b(u, C)` is the basal shear stress

    .. math::
       \\tau_b(u, C) = -C|u|^{1/m - 1}u
    """
    return -m/(m + 1) * inner(tau_b(u, C), u) * dx


def gravity(u=None, h=None, s=None):
    """Return the gravitational part of the ice stream action functional

    The gravitational part of the ice stream action functional is

    .. math::
       E(u) = -\int_\Omega\\rho_Igh\\nabla s\cdot u\hspace{2pt}dx

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness
    s : firedrake.Function
        ice surface elevation
    """
    return -rho_ice * g * h * inner(grad(s), u) * dx


def terminus(u=None, h=None, s=None, ice_front_ids=None):
    """Return the terminal stress part of the ice stream action functional

    The power exerted due to stress at the ice calving terminus :math:`\Gamma`
    is

    .. math::
       E(u) = \int_\Gamma\left(\\frac{1}{2}\\rho_Igh^2 - \\rho_Wgd^2\\right)
       u\cdot \\nu\hspace{2pt}ds

    where :math:`d` is the water depth at the terminus. We assume that sea
    level is at :math:`z = 0` for purposes of calculating the water depth.

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness
    s : firedrake.Function
        ice surface elevation
    ice_front_ids : list of int
        numeric IDs of the parts of the boundary corresponding to the
        calving front
    """
    from firedrake import conditional, lt
    d = conditional(lt(s - h, 0), s - h, 0)

    tau_I = rho_ice * g * h**2 / 2
    tau_W = rho_water * g * d**2 / 2

    mesh = u.ufl_domain()
    n = firedrake.FacetNormal(mesh)

    IDs = tuple(ice_front_ids)
    return (tau_I - tau_W) * inner(u, n) * ds(IDs)


class IceStream(object):
    """Class for modelling the flow of grounded ice streams

    This class provides functions that solve for the velocity, thickness,
    and surface elevation of a grounded, fast-flowing ice stream.

    .. seealso::
       :py:func:`icepack.models.viscosity.viscosity_depth_averaged`
          Default implementation of the ice stream viscous action
    """
    def __init__(self, viscosity=viscosity, friction=friction,
                       gravity=gravity, terminus=terminus):
        self.mass_transport = MassTransport()
        self.viscosity = add_kwarg_wrapper(viscosity)
        self.friction = add_kwarg_wrapper(friction)
        self.gravity = add_kwarg_wrapper(gravity)
        self.terminus = add_kwarg_wrapper(terminus)

    def action(self, u=None, h=None, s=None, ice_front_ids=[], **kwargs):
        """Return the action functional that gives the ice stream diagnostic
        model as the Euler-Lagrange equations"""
        viscosity = self.viscosity(u=u, h=h, s=s, **kwargs)
        friction = self.friction(u=u, h=h, s=s, **kwargs)
        gravity = self.gravity(u=u, h=h, s=s, **kwargs)
        terminus = self.terminus(u=u, h=h, s=s,
                                 ice_front_ids=ice_front_ids, **kwargs)

        return viscosity + friction - gravity - terminus

    def diagnostic_solve(self, u0=None, h=None, s=None,
                         dirichlet_ids=[], tol=1e-6, **kwargs):
        """Solve for the ice velocity from the thickness and surface
        elevation

        Parameters
        ----------
        u0 : firedrake.Function
            Initial guess for the ice velocity; the Dirichlet boundaries
            are taken from `u0`
        h : firedrake.Function
            Ice thickness
        s : firedrake.Function
            Ice surface elevation
        dirichlet_ids : list of int
            list of integer IDs denoting the parts of the boundary where
            Dirichlet boundary conditions should be applied
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
            `viscosity`, `friction`, `gravity`, and `terminus` functions
            that were set when this model object was initialized
        """
        u = u0.copy(deepcopy=True)
        viscosity = self.viscosity(u=u, h=h, s=s, **kwargs)
        friction = self.friction(u=u, **kwargs)
        scale = firedrake.assemble(viscosity + friction)
        tolerance = tol * scale

        mesh = u.ufl_domain()
        boundary_ids = list(mesh.topology.exterior_facets.unique_markers)
        IDs = list(set(boundary_ids) - set(dirichlet_ids))
        bcs = [firedrake.DirichletBC(u.function_space(), (0, 0), k)
               for k in dirichlet_ids]

        action = self.action(u=u, h=h, s=s, ice_front_ids=IDs, **kwargs)
        return newton_search(action, u, bcs, tolerance)

    def prognostic_solve(self, dt, h0=None, a=None, u=None, **kwargs):
        """Propagate the ice thickness forward one timestep
        """
        return self.mass_transport.solve(dt, h0=h0, a=a, u=u, **kwargs)

    def compute_surface(self, h=None, b=None):
        """Return the ice surface elevation consistent with a given
        thickness and bathymetry

        If the bathymetry beneath a tidewater glacier is too low, the ice
        will go afloat. The surface elevation of a floating ice shelf is

        .. math::
           s = (1 - \rho_I / \rho_W)h,

        provided everything is in hydrostatic balance.
        """
        Q = h.ufl_function_space()
        s_expr = firedrake.max_value(h + b, (1 - rho_ice / rho_water) * h)
        return firedrake.interpolate(s_expr, Q)
