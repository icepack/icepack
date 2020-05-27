# Copyright (C) 2019-2020 by Daniel Shapero <shapero@uw.edu>
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

import warnings
import functools
import sympy
import firedrake
from firedrake import (inner, outer, sym, Identity, tr as trace, sqrt,
                       grad, dx, ds, ds_b, ds_v)
from icepack.models.friction import (bed_friction, side_friction,
                                     normal_flow_penalty)
from icepack.models.mass_transport import LaxWendroff
from icepack.optimization import MinimizationProblem, NewtonSolver
from icepack.constants import (ice_density as ρ_I, water_density as ρ_W,
                               glen_flow_law as n, weertman_sliding_law as m,
                               gravity as g)
from icepack.utilities import (facet_normal_2, grad_2, diameter,
                               add_kwarg_wrapper,
                               compute_surface as _compute_surface)

def gravity(u, h, s):
    r"""Return the gravitational part of the ice stream action functional

    The gravitational part of the hybrid model action functional is

    .. math::
       E(u) = -\int_\Omega\int_0^1\rho_Ig\nabla s\cdot u\; h\, d\zeta\; dx

    Parameters
    ----------
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness
    s : firedrake.Function
        ice surface elevation
    """
    return -ρ_I * g * inner(grad_2(s), u) * h


def _legendre(n, ζ):
    return sympy.functions.special.polynomials.legendre(n, 2 * ζ - 1)


@functools.lru_cache(maxsize=None)
def _pressure_approx(N):
    ζ, ζ_sl = sympy.symbols('ζ ζ_sl', real=True, positive=True)

    def coefficient(n):
        Sn = _legendre(n, ζ)
        norm_square = sympy.integrate(Sn**2, (ζ, 0, 1))
        return sympy.integrate((ζ_sl - ζ) * Sn, (ζ, 0, ζ_sl)) / norm_square

    polynomial = sum([coefficient(n) * _legendre(n, ζ) for n in range(N)])
    return sympy.lambdify((ζ, ζ_sl), sympy.simplify(polynomial))


def terminus(u, h, s):
    r"""Return the terminal stress part of the hybrid model action functional

    The power exerted due to stress at the calving terminus :math:`\Gamma` is

    .. math::
        E(u) = \int_\Gamma\int_0^1\left(\rho_Ig(1 - \zeta) -
        \rho_Wg(\zeta_{\text{sl}} - \zeta)_+\right)u\cdot\nu\; h\, d\zeta\; ds

    where :math:`\zeta_\text{sl}` is the relative depth to sea level and the
    :math:`(\zeta_\text{sl} - \zeta)_+` denotes only the positive part.

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
    mesh = u.ufl_domain()
    zdegree = u.ufl_element().degree()[1]

    x, y, ζ = firedrake.SpatialCoordinate(mesh)
    b = s - h
    ζ_sl = firedrake.max_value(-b, 0) / h
    p_W = ρ_W * g * h * _pressure_approx(zdegree + 1)(ζ, ζ_sl)
    p_I = ρ_I * g * h * (1 - ζ)

    ν = facet_normal_2(mesh)
    return (p_I - p_W) * inner(u, ν) * h


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
        E(u) = \frac{n}{n + 1}\int_\Omega\int_0^1\left(
        M : \dot\varepsilon_x + \tau_z\cdot\varepsilon_z\right)h\, d\zeta\; dx

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
    return n / (n + 1) * (inner(M, ε_x) + inner(τ_z, ε_z)) * h


class HybridModel(object):
    r"""Class for modeling 3D glacier flow velocity

    This class provides functions that solve for the thickness, surface
    elevation, and 3D velocity of glaciers of arbitrary speed and flow
    regime (fast-sliding or no sliding). This model assumes that the domain
    is extruded from a 2D footprint mesh. Moreover, the mesh is assumed to
    have a uniform thickness of 1, i.e. it has not been stretch to the bed
    and surface topography.
    """
    def __init__(self, viscosity=viscosity, friction=bed_friction,
                 gravity=gravity, terminus=terminus,
                 mass_transport=LaxWendroff(dimension=3)):
        self.mass_transport = mass_transport
        self.viscosity = add_kwarg_wrapper(viscosity)
        self.friction = add_kwarg_wrapper(friction)
        self.side_friction = add_kwarg_wrapper(side_friction)
        self.gravity = add_kwarg_wrapper(gravity)
        self.terminus = add_kwarg_wrapper(terminus)
        self.penalty = add_kwarg_wrapper(normal_flow_penalty)

    def action(self, u, h, s, **kwargs):
        r"""Return the action functional that gives the hybrid model as its
        Euler-Lagrange equations"""
        mesh = u.ufl_domain()
        ice_front_ids = tuple(kwargs.pop('ice_front_ids', ()))
        side_wall_ids = tuple(kwargs.pop('side_wall_ids', ()))

        viscosity = self.viscosity(u=u, h=h, s=s, **kwargs) * dx
        gravity = self.gravity(u=u, h=h, s=s, **kwargs) * dx

        friction = self.friction(u=u, h=h, s=s, **kwargs) * ds_b

        ds_w = ds_v(domain=mesh, subdomain_id=side_wall_ids)
        side_friction = self.side_friction(u=u, h=h, s=s, **kwargs) * ds_w
        penalty = self.penalty(u=u, h=h, s=s, **kwargs) * ds_w

        xdegree_u, zdegree_u = u.ufl_element().degree()
        degree_h = h.ufl_element().degree()[0]
        degree = (xdegree_u + degree_h, 2 * zdegree_u + 1)
        metadata = {'quadrature_degree': degree}
        ds_t = ds_v(domain=mesh, subdomain_id=ice_front_ids, metadata=metadata)
        terminus = self.terminus(u=u, h=h, s=s, **kwargs) * ds_t

        return (viscosity + friction + side_friction
                - gravity - terminus + penalty)

    def scale(self, u, h, s, **kwargs):
        r"""Return the positive, convex part of the action functional

        The positive part of the action functional is used as a dimensional
        scale to determine when to terminate an optimization algorithm.
        """
        return (self.viscosity(u=u, h=h, s=s, **kwargs) * dx +
                self.friction(u=u, h=h, s=s, **kwargs) * ds_b)

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
        problem = MinimizationProblem(action, scale, u, bcs, params)
        solver = NewtonSolver(problem, tol)
        solver.solve()
        return u

    def prognostic_solve(self, dt, h0, a, u, **kwargs):
        return self.mass_transport.solve(dt, h0=h0, a=a, u=u, **kwargs)

    def compute_surface(self, h, b):
        warnings.warn('Compute surface moved from member function of models to'
                      ' icepack module; call `icepack.compute_surface` instead'
                      ' of e.g. `ice_stream.compute_surface`',
                      DeprecationWarning)
        return _compute_surface(h, b)
