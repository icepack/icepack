# Copyright (C) 2017-2020 by Daniel Shapero <shapero@uw.edu>
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

r"""Miscellaneous utilities for depth-averaging 3D fields, computing
horizontal gradients of 3D fields, lifting 2D fields into 3D, etc."""

import warnings
import functools
import inspect
import sympy
import numpy as np
import firedrake
from firedrake import sqrt, tr, det
from icepack.constants import ice_density as ρ_I, water_density as ρ_W


default_solver_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}


def get_kwargs_alt(dictionary, keys, keys_alt):
    r"""Get value from dictionary by key or by an alternate, deprecated key

    This helper function is to aid in a refactoring of icepack where shorter
    keyword arguments were replaced by longer names, for example `velocity`
    instead of `u`, `thickness` instead of `h`, etc. For backwards
    compatibility, it should be possible to use either the new or old keyword
    argument names, but a warning should be thrown on using the old name.
    """
    if all([key in dictionary for key in keys]):
        return map(dictionary.__getitem__, keys)

    warnings.warn(f"Abbreviated names {keys_alt} have been deprecated, use "
                  f"full names {keys} instead.", FutureWarning, stacklevel=2)
    return tuple((dictionary.get(key, dictionary.get(alt_key))
                  for key, alt_key in zip(keys, keys_alt)))


def _legendre(n, ζ):
    return sympy.functions.special.polynomials.legendre(n,2 * ζ -1)

def facet_normal_2(mesh):
    r"""Compute the horizontal component of the unit outward normal vector
    to a mesh"""
    ν = firedrake.FacetNormal(mesh)
    return firedrake.as_vector((ν[0], ν[1]))


def grad_2(q):
    r"""Compute the horizontal gradient of a 3D field"""
    return firedrake.as_tensor((q.dx(0), q.dx(1)))


def div_2(q):
    r"""Compute the horizontal divergence of a 3D field"""
    return q[0].dx(0) + q[1].dx(1)


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
    h = kwargs.get('thickness', kwargs.get('h'))
    b = kwargs.get('bed', kwargs.get('b'))

    Q = h.ufl_function_space()
    s_expr = firedrake.max_value(h + b, (1 - ρ_I / ρ_W) * h)
    return firedrake.interpolate(s_expr, Q)


def depth_average(q3d, weight=firedrake.Constant(1)):
    r"""Return the weighted depth average of a function on an extruded mesh"""
    element3d = q3d.ufl_element()

    # Create the element `E x DG0` where `E` is the horizontal element for the
    # input field
    element_z = firedrake.FiniteElement(family='DG', cell='interval', degree=0)
    shape = q3d.ufl_shape
    if len(shape) == 0:
        element_xy = element3d.sub_elements()[0]
        element_avg = firedrake.TensorProductElement(element_xy, element_z)
        element2d = element_xy
    elif len(shape) == 1:
        element_xy = element3d.sub_elements()[0].sub_elements()[0]
        element_u = firedrake.TensorProductElement(element_xy, element_z)
        element_avg = firedrake.VectorElement(element_u, dim=shape[0])
        element2d = firedrake.VectorElement(element_xy, dim=shape[0])
    else:
        raise NotImplementedError('Depth average of tensor fields not yet '
                                  'implemented!')

    # Project the weighted 3D field onto vertical DG0
    mesh3d = q3d.ufl_domain()
    Q_avg = firedrake.FunctionSpace(mesh3d, element_avg)
    q_avg = firedrake.project(weight * q3d, Q_avg)

    # Create a function space on the 2D mesh and a 2D function defined on this
    # space, then copy the raw vector of expansion coefficients from the 3D DG0
    # field into the coefficients of the 2D field. TODO: Get some assurance
    # from the firedrake folks that this will always work.
    mesh2d = mesh3d._base_mesh
    Q2D = firedrake.FunctionSpace(mesh2d, element2d)
    q2d = firedrake.Function(Q2D)
    q2d.dat.data[:] = q_avg.dat.data_ro[:]

    return q2d


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


@functools.lru_cache(maxsize=None)
def vertically_integrate(q,h):
    r"""
    q : firedrake.Function
        integrand
    h : firedrake.Function
        ice thickness
    """
    def weight(n,ζ):
        norm=(1/sympy.integrate(_legendre(n,ζ)**2,(ζ,0,1)))**.5
        return sympy.lambdify(ζ,norm*_legendre(n,ζ),'numpy')

    def coefficient(n,q,ζ,ζsym,Q):
        a_n=depth_average(q,weight=weight(n,ζsym)(ζ))
        a_n3d=lift3d(a_n,Q)
        return a_n3d

    def recurrance_relation(n,ζ):
        if n>0:
            return sympy.lambdify(ζ,(1/(2*(2*n+1)))*(_legendre(n+1,ζ)-_legendre(n-1,ζ)),'numpy')
        elif n==0:
            return sympy.lambdify(ζ,ζ,'numpy')
        if n<0:
            raise ValueError("n must be positive")

    Q=h.function_space()
    mesh=Q.mesh()
    x,y,ζ=firedrake.SpatialCoordinate(mesh)
    xdegree_q,zdegree_q=q.ufl_element().degree()

    ζsym = sympy.symbols('ζsym', real=True, positive=True)

    q_int=sum([coefficient(k,q,ζ,ζsym,Q) * recurrance_relation(k,ζsym)(ζ) for k in range(zdegree_q)])
    return q_int

def vertical_velocity(u,h,m=0.0):
    r"""
    u : firedrake.Function
        ice velocity
    h : firedrake.Function
        ice thickness
    m : firedrake.Function
        basal vertical velocity
    """
    Q = h.function_space()
    mesh = Q.mesh()
    xdegree_u, zdegree_u = u.ufl_element().degree()
    W = firedrake.FunctionSpace(mesh,family='CG',degree=xdegree_u,vfamily='GL',vdegree=zdegree_u)
    u_div = firedrake.interpolate(u[0].dx(0)+u[1].dx(1),W)
    return (m/h-vertically_integrate(u_div,h))


def add_kwarg_wrapper(func):
    signature = inspect.signature(func)
    if any(str(signature.parameters[param].kind) == 'VAR_KEYWORD'
           for param in signature.parameters):
        return func

    params = signature.parameters

    def wrapper(*args, **kwargs):
        kwargs_ = dict((key, kwargs[key]) for key in kwargs if key in params)
        return func(*args, **kwargs_)

    return wrapper
