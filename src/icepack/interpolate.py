# Copyright (C) 2017-2026 by Daniel Shapero <shapero@uw.edu> and David
# Lilien
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

r"""Functions for interpolating gridded remote sensing data sets to finite
element spaces"""

from functools import singledispatch
from collections.abc import Sequence
import numpy as np
import ufl
import rasterio
import xarray
from scipy.interpolate import RegularGridInterpolator
import firedrake
from firedrake import (
    Constant, inner, dot, grad, dx, ds, dS, avg, jump, action, adjoint, derivative
)
from petsc4py import PETSc


@singledispatch
def _sample(dataset, X, **kwargs):
    raise TypeError(
        "Input must be a single or sequence of `rasterio.DatasetReader` or "
        "`xarray.DataArray`!"
    )


@_sample.register
def _sample_rasterio_scalar(dataset: rasterio.DatasetReader, X, **kwargs):
    xres = dataset.res[0]
    yres = dataset.res[1]
    bounds = dataset.bounds
    xmin = max(X[:, 0].min() - 3 * xres, bounds.left)
    xmax = min(X[:, 0].max() + 3 * xres, bounds.right)
    ymin = max(X[:, 1].min() - 3 * yres, bounds.bottom)
    ymax = min(X[:, 1].max() + 3 * yres, bounds.top)

    window = rasterio.windows.from_bounds(
        left=xmin,
        right=xmax,
        bottom=ymin,
        top=ymax,
        transform=dataset.transform,
    )
    window = window.round_lengths(op="ceil").round_offsets(op="floor")
    transform = rasterio.windows.transform(window, dataset.transform)

    upper_left = transform * (0, 0)
    lower_right = transform * (window.width - 1, window.height - 1)
    xs = np.linspace(upper_left[0], lower_right[0], window.width)
    ys = np.linspace(lower_right[1], upper_left[1], window.height)

    data = np.flipud(dataset.read(indexes=1, window=window, masked=True)).T
    method = kwargs.get("method", "linear")
    interpolator = RegularGridInterpolator((xs, ys), data, method=method)
    return interpolator(X, method=method)


@_sample.register
def _xarray_sample(dataset: xarray.DataArray, X, **kwargs):
    x = xarray.DataArray(X[:, 0], dims="z")
    y = xarray.DataArray(X[:, 1], dims="z")
    method = kwargs.get("method", "linear")
    return dataset.interp(x=x, y=y, method=method).to_numpy()


@_sample.register
def _sample_vector(f: Sequence, X, **kwargs):
    return np.column_stack([_sample(fi, X, **kwargs) for fi in f])


def interpolate(f, Q, **kwargs):
    r"""Interpolate an expression or a gridded data set to a function space

    Parameters
    ----------
    f : rasterio dataset or tuple of rasterio datasets
        The gridded data set for scalar fields or the tuple of gridded data
        sets for each component
    Q : firedrake.FunctionSpace
        The function space where the result will live

    Returns
    -------
    firedrake.Function
        A finite element function defined on `Q` with the same nodal values
        as the data `f`
    """
    if isinstance(f, (ufl.core.expr.Expr, firedrake.Function)):
        return firedrake.Function(Q).interpolate(f)

    mesh = Q.mesh()
    element = Q.ufl_element()

    # Cannot take sub-elements if function is 3D scalar, otherwise shape will
    # mismatch vertical basis. This attempts to distinguish if multiple
    # subelements due to dimension or vector function.
    if issubclass(type(element), firedrake.VectorElement):
        # NOTE: UFL changed getting sub-elements from a function to a property
        # so we have some try/except hackery to make this work for old and new
        # versions.
        try:
            element = element.sub_elements()[0]
        except TypeError:
            element = element.sub_elements[0]

    V = firedrake.VectorFunctionSpace(mesh, element)
    X = firedrake.Function(V).interpolate(mesh.coordinates).dat.data_ro[:, :2]

    q = firedrake.Function(Q)
    q.dat.data[:] = _sample(f, X, **kwargs)
    return q


def fit(data, stddev, smoothing_length, Q, **kwargs):
    r"""Fit a data set to a function defined on some mesh. The data do not
    have to be dense. The fit will not (in general) be exact.

    Parameters
    ----------
    data : firedrake.Function
        The observational data, defined on a VertexOnlyMesh
    stddev : firedrake.Function or firedrake.Constant
        The standard deviation of the measurement errors, defined on the same
        point cloud as the observational data.
        Should have the same physical units as the data themselves.
    smoothing_length : float
        A length scale determining how far to smooth the fitted field
    Q : firedrake.FunctionSpace
        The function space where the resulting field should live. Must be
        defined on a triangular mesh with Lagrange elements.

    Returns
    -------
    firedrake.Function
        A finite element function defined on `Q`

    Notes
    -----
    This is an experimental feature which only just barely works with some
    low-level hackery. It will be overhauled pending some changes to Firedrake.
    Use at your own risk.
    """
    mesh = Q.mesh()
    if str(cell := mesh.ufl_cell()) != "triangle":
        raise NotImplementedError(
            f"Can't do fitting on {cell} meshes, only triangle!"
        )

    element = Q.ufl_element()
    if (family := element.family()) != "Lagrange":
        raise NotImplementedError(
            f"Can't do fitting into {cell} elements, only Lagrange!"
        )

    hhj = firedrake.FiniteElement("HHJ", "triangle", element.degree() - 1)
    S = firedrake.FunctionSpace(mesh, hhj)
    Z = S * Q
    z = firedrake.Function(Z)
    s, p = firedrake.split(z)

    # TODO: Check the boundary conditions here. In the limit of large smoothing
    # length, we should get back the least-squares fit of a plane.
    α = Constant(smoothing_length)
    area = firedrake.assemble(Constant(1.0) * dx(domain=mesh))
    Ω = Constant(area)
    n = firedrake.FacetNormal(mesh)
    L_cells = (inner(s, grad(grad(p))) - 0.5 * inner(s, s)) * dx
    L_facets = avg(inner(n, dot(s, n))) * jump(grad(p), n) * dS
    L_boundary = inner(n, dot(s, n)) * inner(grad(p), n) * ds
    L = α**4 / Ω * (L_cells - L_facets - L_boundary)
    A = derivative(derivative(L, z), z)

    # Make the map that interpolates functions on the mesh into the point cloud
    D = data.function_space()
    _, q = firedrake.TrialFunctions(Z)
    I = firedrake.interpolate(q, D)

    # The (inverse) covariance matrix, on the point cloud. Weights each
    # observation by the reciprocal of the variance. The sum is normalized
    # by the total rms variance, which makes the misfit a weighted mean.
    total_precision = firedrake.assemble(1 / stddev**2 * dx(domain=D.mesh()))
    Π = Constant(1 / np.sqrt(total_precision))
    q, r = firedrake.TestFunction(D), firedrake.TrialFunction(D)
    Σ = Π * q * r / stddev**2 * dx

    # The "gain matrix" K does a round trip from the mesh to the point cloud
    # and back. TODO: Patch Firedrake so we don't need this awful hackery
    try:
        kw = {"allocation_integral_types": ("cell",)}
        K = firedrake.assemble(action(adjoint(I), action(Σ, I)), **kw)
    except PETSc.Error:
        raise NotImplementedError(
            "This feature only works for Firedrake versions 2026.4.2 and up."
        )

    H = firedrake.assemble(A + K)
    F = firedrake.assemble(action(adjoint(I), action(Σ, data)))

    default_sparams = {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    sparams = kwargs.get("solver_parameters", default_sparams)
    firedrake.solve(H, z, F, solver_parameters=sparams)
    return z.subfunctions[1]
