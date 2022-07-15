# Copyright (C) 2017-2022 by Daniel Shapero <shapero@uw.edu>
# Andrew Hoffman <hoffmaao@uw.edu>
# Jessica Badgeley <badgeley@uw.edu>
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

r"""Miscellaneous utilities for depth-averaging 3D fields, computing
horizontal gradients of 3D fields, lifting 2D fields into 3D, etc."""

from operator import itemgetter
import inspect
import sympy
import numpy as np
import firedrake
from firedrake import sqrt, tr, det
from .constants import ice_density as ρ_I, water_density as ρ_W
from .calculus import div


default_solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def legendre(n, ζ):
    return sympy.functions.special.polynomials.legendre(n, 2 * ζ - 1)


def eigenvalues(a):
    r"""Return a pair of symbolic expressions for the largest and smallest
    eigenvalues of a 2D rank-2 tensor"""
    tr_a = tr(a)
    det_a = det(a)
    # TODO: Fret about numerical stability
    Δ = sqrt(tr_a**2 - 4 * det_a)
    return ((tr_a + Δ) / 2, (tr_a - Δ) / 2)


def diameter(mesh):
    r"""Compute the diameter of the mesh in the L-infinity metric"""
    X = mesh.coordinates.dat.data_ro
    xmin = mesh.comm.allreduce(np.min(X, axis=0), op=np.minimum)
    xmax = mesh.comm.allreduce(np.max(X, axis=0), op=np.maximum)
    return np.max(xmax - xmin)


def compute_surface(**kwargs):
    r"""Return the ice surface elevation consistent with a given
    thickness and bathymetry

    If the bathymetry beneath a tidewater glacier is too low, the ice
    will go afloat. The surface elevation of a floating ice shelf is

    .. math::
       s = (1 - \rho_I / \rho_W)h,

    provided everything is in hydrostatic balance.
    """
    # TODO: Remove the 'h' and 'b' arguments once these are deprecated.
    h = kwargs.get("thickness", kwargs.get("h"))
    b = kwargs.get("bed", kwargs.get("b"))

    Q = h.ufl_function_space()
    s_expr = firedrake.max_value(h + b, (1 - ρ_I / ρ_W) * h)
    return firedrake.interpolate(s_expr, Q)


def depth_average(q_xz, weight=firedrake.Constant(1)):
    r"""Return the weighted depth average of a function on an extruded mesh"""
    element_xz = q_xz.ufl_element()

    # Create the element `E x DG0` where `E` is the horizontal element for the
    # input field
    element_z = firedrake.FiniteElement(family="R", cell="interval", degree=0)
    shape = q_xz.ufl_shape
    if len(shape) == 0:
        element_x = element_xz.sub_elements()[0]
        element_avg = firedrake.TensorProductElement(element_x, element_z)
    elif len(shape) == 1:
        element_xy = element_xz.sub_elements()[0].sub_elements()[0]
        element_u = firedrake.TensorProductElement(element_xy, element_z)
        element_avg = firedrake.VectorElement(element_u, dim=shape[0])
        element_x = firedrake.VectorElement(element_xy, dim=shape[0])
    else:
        raise NotImplementedError("Depth average of tensor fields not yet implemented!")

    # Project the weighted 3D field onto vertical DG0
    mesh_xz = q_xz.ufl_domain()
    Q_avg = firedrake.FunctionSpace(mesh_xz, element_avg)
    q_avg = firedrake.project(weight * q_xz, Q_avg)

    # Create a function space on the 2D mesh and a 2D function defined on this
    # space, then copy the raw vector of expansion coefficients from the 3D DG0
    # field into the coefficients of the 2D field.
    mesh_x = mesh_xz._base_mesh
    Q_x = firedrake.FunctionSpace(mesh_x, element_x)
    q_x = firedrake.Function(Q_x)
    q_x.dat.data[:] = q_avg.dat.data_ro[:]

    return q_x


def lift3d(q2d, Q3D):
    r"""Return a 3D function that extends the given 2D function as a constant
    in the vertical

    This is the reverse operation of depth averaging -- it takes a 2D function
    `q2d` and returns a function `q3d` defined over a 3D function space such
    `q3d(x, y, z) == q2d(x, y)` for any `x, y`. The space `Q3D` of the result
    must have the same horizontal element as the input function and a vertical
    degree of 0.

    Parameters
    ----------
    q2d : firedrake.Function
        A function defined on a 2D footprint mesh
    Q3D : firedrake.Function
        A function space defined on a 3D mesh extruded from the footprint mesh;
        the function space must only go up to degree 0 in the vertical.

    Returns
    -------
    q3d : firedrake.Function
        The 3D-lifted input field
    """
    q3d = firedrake.Function(Q3D)
    assert q3d.dat.data_ro.shape == q2d.dat.data_ro.shape
    q3d.dat.data[:] = q2d.dat.data_ro[:]
    return q3d


def vertically_integrate(q, h):
    r"""
    q : firedrake.Function
        integrand
    h : firedrake.Function
        ice thickness
    """

    def weight(n, ζ):
        norm = (1 / sympy.integrate(legendre(n, ζ) ** 2, (ζ, 0, 1))) ** 0.5
        return sympy.lambdify(ζ, norm * legendre(n, ζ), "numpy")

    def coefficient(n, q, ζ, ζsym, Q):
        a_n = depth_average(q, weight=weight(n, ζsym)(ζ))
        a_n3d = lift3d(a_n, Q)
        return a_n3d

    def recurrence_relation(n, ζ):
        if n > 0:
            return sympy.lambdify(
                ζ,
                (1 / (2 * (2 * n + 1))) * (legendre(n + 1, ζ) - legendre(n - 1, ζ)),
                "numpy",
            )
        elif n == 0:
            return sympy.lambdify(ζ, ζ, "numpy")
        if n < 0:
            raise ValueError("n must be positive")

    Q = h.function_space()
    mesh = Q.mesh()
    ζ = firedrake.SpatialCoordinate(mesh)[mesh.geometric_dimension() - 1]
    xdegree_q, zdegree_q = q.ufl_element().degree()

    ζsym = sympy.symbols("ζsym", real=True, positive=True)

    q_int = sum(
        [
            coefficient(k, q, ζ, ζsym, Q) * recurrence_relation(k, ζsym)(ζ)
            for k in range(zdegree_q)
        ]
    )
    return q_int


def vertical_velocity(**kwargs):
    r"""Compute the vertical component of the full 3D ice velocity from the
    horizontal components, assuming that the 3D flow field is divergence-free

    Note that the returned vertical velocity is in terrain-following
    coordinates -- it records the relative depth through the ice column and
    thus has units of inverse years.

    Parameters
    ----------
    velocity : firedrake.Function
    thickness : firedrake.Function
    basal_mass_balance : firedrake.Function

    Returns
    -------
    vertical_velocity : firedrake.Function
    """
    u, h, m = itemgetter("velocity", "thickness", "basal_mass_balance")(kwargs)

    Q = h.function_space()
    mesh = Q.mesh()
    xdegree_u, zdegree_u = u.ufl_element().degree()
    W = firedrake.FunctionSpace(mesh, "CG", xdegree_u, vfamily="GL", vdegree=zdegree_u)
    u_div = firedrake.interpolate(div(u), W)
    return m / h - vertically_integrate(u_div, h)


def add_kwarg_wrapper(func):
    signature = inspect.signature(func)
    if any(
        str(signature.parameters[param].kind) == "VAR_KEYWORD"
        for param in signature.parameters
    ):
        return func

    params = signature.parameters

    def wrapper(*args, **kwargs):
        kwargs_ = dict((key, kwargs[key]) for key in kwargs if key in params)
        return func(*args, **kwargs_)

    return wrapper
