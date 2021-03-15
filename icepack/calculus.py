# Copyright (C) 2020-2021 by Daniel Shapero <shapero@uw.edu> and Benjamin Hills
# <benjaminhhills@gmail.com>
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

r"""Vector calculus operators that dispatch on the dimension of the underlying mesh

This module includes functions for calculating things like the gradient or divergence
of a field in a way that does what you mean whether the underlying geometry is a plan-
view, flowband, or 3D model.
"""

import firedrake


def get_mesh_axes(mesh):
    r"""Get a string representing the axes present in the mesh -- 'x', 'xy', 'xz', or
    'xyz'"""
    mesh_dim = mesh.geometric_dimension()
    extruded = mesh.layers is not None
    if mesh_dim == 1:
        return "x"
    if mesh_dim == 2 and extruded:
        return "xz"
    if mesh_dim == 2 and not extruded:
        return "xy"
    if mesh_dim == 3 and extruded:
        return "xyz"

    raise ValueError("icepack requires 3D meshes to be extruded")


def grad(q):
    r"""Compute the gradient of a scalar or vector field"""
    axes = get_mesh_axes(q.ufl_domain())
    if axes == "xy":
        return firedrake.grad(q)
    if axes == "xyz":
        return firedrake.as_tensor((q.dx(0), q.dx(1)))
    return q.dx(0)


def sym_grad(u):
    r"""Compute the symmetric gradient of a vector field"""
    axes = get_mesh_axes(u.ufl_domain())
    if axes == "xy":
        return firedrake.sym(firedrake.grad(u))
    if axes == "xyz":
        return firedrake.sym(firedrake.as_tensor((u.dx(0), u.dx(1))))
    return u.dx(0)


def div(u):
    r"""Compute the horizontal divergence of a velocity field"""
    axes = get_mesh_axes(u.ufl_domain())
    if axes == "xy":
        return firedrake.div(u)
    if axes == "xyz":
        return u[0].dx(0) + u[1].dx(1)
    return u.dx(0)


def FacetNormal(mesh):
    r"""Compute the horizontal component of the unit outward normal vector to a mesh"""
    axes = get_mesh_axes(mesh)
    ν = firedrake.FacetNormal(mesh)
    if axes == "xy":
        return ν
    if axes == "xyz":
        return firedrake.as_vector((ν[0], ν[1]))
    return ν[0]


def trace(A):
    r"""Compute the trace of a rank-2 tensor"""
    axes = get_mesh_axes(A.ufl_domain())
    if axes in ["x", "xz"]:
        return A
    return firedrake.tr(A)


def Identity(dim):
    r"""Return the unit tensor of a given dimension"""
    if dim == 1:
        return firedrake.Constant(1.0)
    return firedrake.Identity(dim)
