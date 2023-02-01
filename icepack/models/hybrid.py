# Copyright (C) 2019-2021 by Daniel Shapero <shapero@uw.edu>
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

import functools
from operator import itemgetter
import sympy
import firedrake
from firedrake import inner, outer, sqrt, dx, ds_b, ds_v
from icepack.models.friction import (
    bed_friction,
    side_friction,
    side_friction_xz,
    normal_flow_penalty,
)
from icepack.models.mass_transport import Continuity
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    glen_flow_law as n,
    gravity as g,
    strain_rate_min,
)
from icepack.utilities import add_kwarg_wrapper, legendre
from icepack.calculus import grad, sym_grad, trace, Identity, FacetNormal, get_mesh_axes


def gravity(**kwargs):
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
    u, h, s = itemgetter("velocity", "thickness", "surface")(kwargs)
    return -ρ_I * g * inner(grad(s), u) * h


@functools.lru_cache(maxsize=None)
def _pressure_approx(N):
    ζ, ζ_sl = sympy.symbols("ζ ζ_sl", real=True, positive=True)

    def coefficient(k):
        Sk = legendre(k, ζ)
        norm_square = sympy.integrate(Sk**2, (ζ, 0, 1))
        return sympy.integrate((ζ_sl - ζ) * Sk, (ζ, 0, ζ_sl)) / norm_square

    polynomial = sum([coefficient(k) * legendre(k, ζ) for k in range(N)])
    return sympy.lambdify((ζ, ζ_sl), sympy.simplify(polynomial))


def terminus(**kwargs):
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
    u, h, s = itemgetter("velocity", "thickness", "surface")(kwargs)

    mesh = u.ufl_domain()
    zdegree = u.ufl_element().degree()[1]

    ζ = firedrake.SpatialCoordinate(mesh)[mesh.geometric_dimension() - 1]

    b = s - h
    ζ_sl = firedrake.max_value(-b, 0) / h
    p_W = ρ_W * g * h * _pressure_approx(zdegree + 1)(ζ, ζ_sl)
    p_I = ρ_I * g * h * (1 - ζ)

    ν = FacetNormal(mesh)
    return (p_I - p_W) * inner(u, ν) * h


def _effective_strain_rate(ε_x, ε_z, ε_min):
    return sqrt((inner(ε_x, ε_x) + trace(ε_x) ** 2 + inner(ε_z, ε_z) + ε_min**2) / 2)


def stresses(**kwargs):
    r"""Calculate the membrane and vertical shear stresses for the given
    horizontal and shear strain rates and fluidity"""
    ε_x, ε_z, A = itemgetter("strain_rate_x", "strain_rate_z", "fluidity")(kwargs)
    ε_min = firedrake.Constant(kwargs.get("strain_rate_min", strain_rate_min))
    ε_e = _effective_strain_rate(ε_x, ε_z, ε_min)
    μ = 0.5 * A ** (-1 / n) * ε_e ** (1 / n - 1)
    I = Identity(ε_x.ufl_domain().geometric_dimension() - 1)
    return 2 * μ * (ε_x + trace(ε_x) * I), 2 * μ * ε_z


def horizontal_strain_rate(**kwargs):
    r"""Calculate the horizontal strain rate with corrections for terrain-
    following coordinates"""
    u, h, s = itemgetter("velocity", "surface", "thickness")(kwargs)
    mesh = u.ufl_domain()
    dim = mesh.geometric_dimension()
    ζ = firedrake.SpatialCoordinate(mesh)[dim - 1]
    b = s - h
    v = -((1 - ζ) * grad(b) + ζ * grad(s)) / h
    du_dζ = u.dx(dim - 1)
    return sym_grad(u) + 0.5 * (outer(du_dζ, v) + outer(v, du_dζ))


def vertical_strain_rate(**kwargs):
    r"""Calculate the vertical strain rate with corrections for terrain-
    following coordinates"""
    u, h = itemgetter("velocity", "thickness")(kwargs)
    mesh = u.ufl_domain()
    du_dζ = u.dx(mesh.geometric_dimension() - 1)
    return 0.5 * du_dζ / h


def viscosity(**kwargs):
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

    Keyword arguments
    -----------------
    velocity : firedrake.Function
    surface : firedrake.Function
    thickness : firedrake.Function
    fluidity : firedrake.Function
        `A` in Glen's flow law

    Returns
    -------
    firedrake.Form
    """
    u, h, s, A = itemgetter("velocity", "thickness", "surface", "fluidity")(kwargs)
    ε_min = kwargs.get("strain_rate_min", firedrake.Constant(strain_rate_min))

    ε_x = horizontal_strain_rate(velocity=u, surface=s, thickness=h)
    ε_z = vertical_strain_rate(velocity=u, thickness=h)
    ε_e = _effective_strain_rate(ε_x, ε_z, ε_min)
    return 2 * n / (n + 1) * h * A ** (-1 / n) * ε_e ** (1 / n + 1)


class HybridModel:
    r"""Class for modeling 3D glacier flow velocity

    This class provides functions that solve for the thickness, surface
    elevation, and 3D velocity of glaciers of arbitrary speed and flow
    regime (fast-sliding or no sliding). This model assumes that the domain
    is extruded from a 2D footprint mesh. Moreover, the mesh is assumed to
    have a uniform thickness of 1, i.e. it has not been stretch to the bed
    and surface topography.
    """

    def __init__(
        self,
        viscosity=viscosity,
        friction=bed_friction,
        gravity=gravity,
        terminus=terminus,
        continuity=Continuity(),
    ):
        self.viscosity = add_kwarg_wrapper(viscosity)
        self.friction = add_kwarg_wrapper(friction)
        self.side_friction = add_kwarg_wrapper(side_friction)
        self.side_friction_xz = add_kwarg_wrapper(side_friction_xz)
        self.gravity = add_kwarg_wrapper(gravity)
        self.terminus = add_kwarg_wrapper(terminus)
        self.penalty = add_kwarg_wrapper(normal_flow_penalty)
        self.continuity = continuity

    def action(self, **kwargs):
        r"""Return the action functional that gives the hybrid model as its
        Euler-Lagrange equations"""
        u, h = itemgetter("velocity", "thickness")(kwargs)
        mesh = u.ufl_domain()
        ice_front_ids = tuple(kwargs.pop("ice_front_ids", ()))
        side_wall_ids = tuple(kwargs.pop("side_wall_ids", ()))

        metadata = {"quadrature_degree": self.quadrature_degree(**kwargs)}
        dx = firedrake.dx(metadata=metadata)
        ds_b = firedrake.ds_b(domain=mesh, metadata=metadata)
        ds_v = firedrake.ds_v(domain=mesh)

        viscosity = self.viscosity(**kwargs) * dx
        gravity = self.gravity(**kwargs) * dx
        friction = self.friction(**kwargs) * ds_b

        if get_mesh_axes(mesh) == "xyz":
            penalty = self.penalty(**kwargs) * ds_v(side_wall_ids)
            side_friction = self.side_friction(**kwargs) * ds_v(side_wall_ids)
        else:
            penalty = 0.0
            side_friction = self.side_friction_xz(**kwargs) * dx

        xdegree_u, zdegree_u = u.ufl_element().degree()
        degree_h = h.ufl_element().degree()[0]
        degree = (xdegree_u + degree_h, 2 * zdegree_u + 1)
        ds_t = firedrake.ds_v(ice_front_ids, metadata={"quadrature_degree": degree})
        terminus = self.terminus(**kwargs) * ds_t

        return viscosity + friction + side_friction - gravity - terminus + penalty

    def scale(self, **kwargs):
        r"""Return the positive, convex part of the action functional

        The positive part of the action functional is used as a dimensional
        scale to determine when to terminate an optimization algorithm.
        """
        u = kwargs["velocity"]
        mesh = u.ufl_domain()
        metadata = {"quadrature_degree": self.quadrature_degree(**kwargs)}
        dx = firedrake.dx(metadata=metadata)
        ds_b = firedrake.ds_b(domain=mesh, metadata=metadata)
        return self.viscosity(**kwargs) * dx + self.friction(**kwargs) * ds_b

    def quadrature_degree(self, **kwargs):
        r"""Return the quadrature degree necessary to integrate the action
        functional accurately

        Firedrake uses a very conservative algorithm for estimating the
        number of quadrature points necessary to integrate a given
        expression. By exploiting known structure of the problem, we can
        reduce the number of quadrature points while preserving accuracy.
        """
        u, h = itemgetter("velocity", "thickness")(kwargs)
        xdegree_u, zdegree_u = u.ufl_element().degree()
        degree_h = h.ufl_element().degree()[0]
        return (
            3 * (xdegree_u - 1) + 2 * degree_h,
            3 * max(zdegree_u - 1, 0) + zdegree_u + 1,
        )
