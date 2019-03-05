# Copyright (C) 2019 by Daniel Shapero <shapero@uw.edu>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

import sympy
import firedrake
from firedrake import inner, outer, sym, Identity, tr as trace, sqrt, \
    grad, dx, ds, ds_b, ds_v
from icepack.constants import rho_ice as ρ_I, rho_water as ρ_W, gravity as g, \
    glen_flow_law as n, weertman_sliding_law as m
from icepack.optimization import newton_search
from icepack.utilities import add_kwarg_wrapper, diameter

def facet_normal_2(mesh):
    r"""Compute the horizontal component of the unit outward normal vector
    to a mesh"""
    ν = firedrake.FacetNormal(mesh)
    return firedrake.as_vector((ν[0], ν[1]))


def grad_2(q):
    r"""Compute the horizontal gradient of a 3D field"""
    return firedrake.as_tensor((q.dx(0), q.dx(1)))


class MassTransport(object):
    def solve(self, dt, h0, a, u, h_inflow=None):
        h_inflow = h_inflow if h_inflow is not None else h0

        Q = h0.function_space()
        h, φ = firedrake.TrialFunction(Q), firedrake.TestFunction(Q)

        ν = facet_normal_2(Q.mesh())
        outflow = firedrake.max_value(inner(u, ν), 0)
        inflow = firedrake.min_value(inner(u, ν), 0)

        flux_cells = -h * inner(u, grad_2(φ)) * dx
        flux_out = h * φ * outflow * ds_v
        F = h * φ * dx + dt * (flux_cells + flux_out)

        accumulation = a * φ * dx
        flux_in = -h_inflow * φ * inflow * ds_v
        A = h0 * φ * dx + dt * (accumulation + flux_in)

        h = h0.copy(deepcopy=True)
        solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
        firedrake.solve(F == A, h, solver_parameters=solver_parameters)

        return h


def gravity(u, h, s):
    r"""Return the gravitational part of the ice stream action functional

    The gravitational part of the hybrid model action functional is

    .. math::
       E(u) = -\int_\Omega\int_0^1\\rho_Ig\\nabla s\cdot u
       \hspace{2pt}hd\zeta \hspace{2pt}dx

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness
    s : firedrake.Function
        ice surface elevation
    """
    return -ρ_I * g * inner(grad_2(s), u) * h * dx


def _legendre(n, ζ):
    return sympy.functions.special.polynomials.legendre(n, 2 * ζ - 1)


def _pressure_approx(N):
    ζ, ζ_sl = sympy.symbols('ζ ζ_sl', real=True, positive=True)

    def coefficient(n):
        Sn = _legendre(n, ζ)
        norm_square = sympy.integrate(Sn**2, (ζ, 0, 1))
        return sympy.integrate((ζ_sl - ζ) * Sn, (ζ, 0, ζ_sl)) / norm_square

    polynomial = sum([coefficient(n) * _legendre(n, ζ) for n in range(N)])
    return sympy.lambdify((ζ, ζ_sl), sympy.simplify(polynomial))


def terminus(u, h, s, ice_front_ids=()):
    xdegree_u, zdegree_u = u.ufl_element().degree()
    degree_h = h.ufl_element().degree()[0]
    degree = (xdegree_u + degree_h, 2 * zdegree_u + 1)
    metadata = {'quadrature_degree': degree}

    x, y, ζ = firedrake.SpatialCoordinate(u.ufl_domain())
    b = s - h
    ζ_sl = firedrake.max_value(-b, 0) / h
    p_W = ρ_W * g * h * _pressure_approx(zdegree_u + 1)(ζ, ζ_sl)
    p_I = ρ_I * g * h * (1 - ζ)

    ν = facet_normal_2(u.ufl_domain())
    dγ = ds_v(tuple(ice_front_ids), metadata=metadata)
    return (p_I - p_W) * inner(u, ν) * h * dγ


def stresses(ε_x, ε_z, A):
    r"""Calculate the membrane and vertical shear stresses for the given
    horizontal and shear strain rates and fluidity"""
    I = Identity(2)
    tr = trace(ε_x)
    ε_e = sqrt((inner(ε_x, ε_x) + inner(ε_z, ε_z) + tr**2) / 2)
    μ = 0.5 * A**(-1/n) * ε_e**(1/n - 1)
    return 2 * μ * (ε_x + tr * I), 2 * μ * ε_z


def horizontal_strain(u, s, h):
    r"""Calculate the horizontal strain rate with corrections for terrain-
    following coordinates"""
    x, y, ζ = firedrake.SpatialCoordinate(u.ufl_domain())
    b = s - h
    v = -((1 - ζ) * grad_2(b) + ζ * grad_2(s)) / h
    du_dζ = u.dx(2)
    return sym(grad_2(u)) + 0.5 * (outer(du_dζ, v) + outer(v, du_dζ))


def vertical_strain(u, h):
    r"""Calculate the vertical strain rate with corrections for terrain-
    following coordinates"""
    du_dζ = u.dx(2)
    return 0.5 * du_dζ / h


def viscosity(u, s, h, A):
    r"""Return the viscous part of the hybrid model action functional

    The viscous component of the action for the hybrid model is

    .. math::
        E(u) = \\frac{n}{n + 1}\int\_\Omega\int_0^1\left(
        M : \dot\varepsilon_x + \tau_z\cdot\varepsilon_z\right)
        hd\zeta\hspace{2pt}dx

    where :math:`M(\dot\varepsilon, A)` is the membrane stress tensor and
    :math:`\tau_z` is the vertical shear stress vector.

    This form assumes that we're using the fluidity parameter instead
    the rheology parameter, the temperature, etc. To use a different
    variable, you can implement your own viscosity functional and pass it
    as an argument when initializing model objects to use your functional
    instead.

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    s : firedrake.Function
        ice surface elevation
    h : firedrake.Function
        ice thickness
    A : firedrake.Function
        ice fluidity parameter

    Returns
    -------
    firedrake.Form
    """
    ε_x, ε_z = horizontal_strain(u, s, h), vertical_strain(u, h)
    M, τ_z = stresses(ε_x, ε_z, A)
    return n / (n + 1) * (inner(M, ε_x) + inner(τ_z, ε_z)) * h * dx


def friction_stress(u, C):
    r"""Compute the shear stress for a given sliding velocity"""
    return -C * sqrt(inner(u, u))**(1/m - 1) * u


def bed_friction(u, C):
    r"""Return the bed friction part of the ice stream action functional

    The frictional part of the ice stream action functional is

    .. math::
       E(u) = -\\frac{m}{m + 1}\int_\Omega\\tau(u, C)\cdot u\hspace{2pt}dx,

    where everything is evaluated at the ice base (:math:`\zeta = 0`) and
    :math:`\\tau(u, C)` is the basal shear stress

    .. math::
       \\tau(u, C) = -C|u|^{1/m - 1}u
    """
    τ_b = friction_stress(u, C)
    return -m/(m + 1) * inner(τ_b, u) * ds_b


def side_friction(u, h, Cs=firedrake.Constant(0), side_wall_ids=()):
    r"""Return the side wall friction part of the action functional

    The component of the action functional due to friction along the side
    walls of the domain is

    .. math::
       E(u) = -\\frac{m}{m + 1}\int_\Gamma\int_0^1 h\\tau(u, C_s)\cdot u
       \hspace{2pt}hd\zeta\hspace{2pt}ds

    where :math:`\\tau(u, C_s)` is the side wall shear stress, :math:`ds`
    is the element of surface area and :math:`\\Gamma` are the side walls.
    Side wall friction is relevant for glaciers that flow through a fjord
    with rock walls on either side.
    """
    mesh = u.ufl_domain()
    ν = facet_normal_2(mesh)
    u_t = u - inner(u, ν) * ν
    τ = friction_stress(u_t, Cs)
    ids = tuple(side_wall_ids)
    return -m/(m + 1) * inner(τ, u_t) * h * ds_v(domain=mesh, subdomain_id=ids)


def normal_flow_penalty(u, h, scale=1.0, exponent=None, side_wall_ids=()):
    r"""Return the penalty for flow normal to the domain boundary

    For problems where a glacier flows along some boundary, e.g. a fjord
    wall, the velocity has to be parallel to this boundary. Rather than
    enforce this boundary condition directly, we add a penalty for normal
    flow to the action functional.
    """
    mesh = u.ufl_domain()
    ν = facet_normal_2(mesh)

    # Note that this quantity has units of [length] x [dimensionless] because
    # the mesh has a "thickness" of 1! If it had dimensions of physical
    # thickness, we would instead use the square root of the facet area.
    δx = firedrake.FacetArea(mesh)
    L = diameter(mesh)

    d = u.ufl_function_space().ufl_element().degree()[0]
    exponent = d + 1 if (exponent is None) else exponent
    penalty = scale * (L / δx)**exponent
    return 0.5 * penalty * inner(u, ν)**2 * h * ds_v(tuple(side_wall_ids))


class HybridModel(object):
    r"""Class for modeling 3D glacier flow velocity

    This class provides functions that solve for the thickness, surface
    elevation, and 3D velocity of glaciers of arbitrary speed and flow
    regime (fast-sliding or no sliding).
    """
    def __init__(self, viscosity=viscosity,
                 friction=bed_friction,
                 gravity=gravity, terminus=terminus):
        self.mass_transport = MassTransport()
        self.viscosity = add_kwarg_wrapper(viscosity)
        self.friction = add_kwarg_wrapper(friction)
        self.side_friction = add_kwarg_wrapper(side_friction)
        self.gravity = add_kwarg_wrapper(gravity)
        self.terminus = add_kwarg_wrapper(terminus)
        self.penalty = add_kwarg_wrapper(normal_flow_penalty)

    def action(self, u, h, s, **kwargs):
        r"""Return the action functional that gives the hybrid model as its
        Euler-Lagrange equations"""
        viscosity = self.viscosity(u=u, h=h, s=s, **kwargs)
        friction = self.friction(u=u, h=h, s=s, **kwargs)
        side_friction = self.side_friction(u=u, h=h, s=s, **kwargs)
        gravity = self.gravity(u=u, h=h, s=s, **kwargs)
        terminus = self.terminus(u=u, h=h, s=s, **kwargs)
        penalty = self.penalty(u=u, h=h, s=s, **kwargs)

        return (viscosity + friction + side_friction
                - gravity - terminus + penalty)

    def scale(self, u, h, s, **kwargs):
        r"""Return the positive, convex part of the action functional

        The positive part of the action functional is used as a dimensional
        scale to determine when to terminate an optimization algorithm.
        """
        return (self.viscosity(u=u, h=h, s=s, **kwargs) +
                self.friction(u=u, h=h, s=s, **kwargs))

    def quadrature_degree(self, u, h, **kwargs):
        r"""Return the quadrature degree necessary to integrate the action
        functional accurately

        Firedrake uses a very conservative algorithm for estimating the
        number of quadrature points necessary to integrate a given
        expression. By exploiting known structure of the problem, we can
        reduce the number of quadrature points while preserving accuracy.
        """
        xdegree_u, zdegree_u = u.ufl_element().degree()
        degree_h = h.ufl_element().degree()[0]
        return (3 * (xdegree_u - 1) + 2 * degree_h,
                3 * max(zdegree_u - 1, 0) + zdegree_u + 1)

    def diagnostic_solve(self, u0, h, s, dirichlet_ids, tol=1e-6, **kwargs):
        r"""Solve for the ice velocity from the thickness and surface
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

        boundary_ids = u.ufl_domain().exterior_facets.unique_markers
        side_wall_ids = kwargs.get('side_wall_ids', [])
        kwargs['side_wall_ids'] = side_wall_ids
        kwargs['ice_front_ids'] = list(
            set(boundary_ids) - set(dirichlet_ids) - set(side_wall_ids))
        bcs = firedrake.DirichletBC(
            u.function_space(), firedrake.as_vector((0, 0)), dirichlet_ids)
        params = {'quadrature_degree': self.quadrature_degree(u, h, **kwargs)}

        action = self.action(u=u, h=h, s=s, **kwargs)
        scale = self.scale(u=u, h=h, s=s, **kwargs)
        return newton_search(action, u, bcs, tol, scale,
                             form_compiler_parameters=params)

    def prognostic_solve(self, dt, h0, a, u, **kwargs):
        return self.mass_transport.solve(dt, h0=h0, a=a, u=u, **kwargs)

    def compute_surface(self, h, b):
        r"""Return the ice surface elevation consistent with a given
        thickness and bathymetry

        If the bathymetry beneath a tidewater glacier is too low, the ice
        will go afloat. The surface elevation of a floating ice shelf is

        .. math::
           s = (1 - \\rho_I / \\rho_W)h,

        provided everything is in hydrostatic balance.
        """
        Q = h.ufl_function_space()
        s_expr = firedrake.max_value(h + b, (1 - ρ_I / ρ_W) * h)
        return firedrake.interpolate(s_expr, Q)
