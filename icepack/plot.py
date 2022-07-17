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

r"""Utilities for plotting gridded data, meshes, and finite element fields

This module contains thin wrappers around functionality in matplotlib for
plotting things. For example, `triplot`, `tricontourf`, and `streamplot`
all call the equivalent matplotlib functions under the hood for plotting
respectively an unstructured mesh, a scalar field on an unstructured mesh,
and a vector field on an unstructured mesh.

The return types should be the same as for the corresponding matplotlib
function. Usually the return type is something that inherits from the
class `ScalarMappable` so that you can make a colorbar out of it.
"""

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm as mpl_cmaps
from matplotlib import streamplot as mpl_streamplot
from matplotlib.ticker import ScalarFormatter
from matplotlib.collections import LineCollection
import numpy as np
import scipy.spatial
import firedrake
import icepack


def _get_coordinates(mesh):
    r"""Return the coordinates of a mesh if the mesh is piecewise linear,
    or interpolate them to piecewise linear if the mesh is curved"""
    coordinates = mesh.coordinates
    element = coordinates.function_space().ufl_element()
    if element.degree() != 1:
        from firedrake import VectorFunctionSpace, interpolate

        V = VectorFunctionSpace(mesh, element.family(), 1)
        coordinates = interpolate(coordinates, V)

    return coordinates


def subplots(*args, **kwargs):
    subplot_kw = kwargs.get("subplot_kw", {})
    subplot_kw["adjustable"] = subplot_kw.get("adjustable", "box")
    kwargs["subplot_kw"] = subplot_kw
    fig, axes = plt.subplots(*args, **kwargs)

    def fmt(ax):
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=True))
        ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

    try:
        if len(axes.shape) == 1:
            for ax in axes:
                fmt(ax)
        else:
            for row in axes:
                for ax in row:
                    fmt(ax)
    except AttributeError:
        fmt(axes)

    return fig, axes


def triplot(mesh, *args, **kwargs):
    r"""Plot a mesh with a different color for each boundary segment"""
    return firedrake.triplot(mesh, *args, **kwargs)


def _project_to_2d(function):
    mesh = function.ufl_domain()
    return function if mesh.layers is None else icepack.depth_average(function)


def tricontourf(function, *args, **kwargs):
    r"""Create a filled contour plot of a finite element field"""
    return firedrake.tricontourf(_project_to_2d(function), *args, **kwargs)


def tricontour(function, *args, **kwargs):
    r"""Create a contour plot of a finite element field"""
    return firedrake.tricontour(_project_to_2d(function), *args, **kwargs)


def tripcolor(function, *args, **kwargs):
    r"""Create a pseudo-color plot of a finite element field"""
    return firedrake.tripcolor(_project_to_2d(function), *args, **kwargs)


def quiver(function, *args, **kwargs):
    r"""Make a quiver plot of a vector field"""
    return firedrake.quiver(_project_to_2d(function), *args, **kwargs)


def streamline(velocity, initial_point, resolution, max_num_points=np.inf):
    r"""Return a streamline of a 2D velocity field

    A streamline :math:`\gamma` of a velocity field :math:`v` is a curve
    that solves the ordinary differential equation

    .. math::
       \frac{d\gamma}{dt} = v(\gamma)

    This function returns an approximate streamline for a velocity field.
    Streamlines are primarily for plotting vector fields but are useful for
    other kinds of postprocessing of vector field data.

    Parameters
    ----------
    velocity : firedrake.Function
        the velocity field we are integrating
    initial_point : pair of floats
        the starting point for the streamline
    resolution : float
        the desired length of each segment of the streamline
    max_num_points : int
        maximum number of points of the streamline; can be necessary to set
        if the trajectory can spiral around a center node

    Returns
    -------
    xs : numpy array of points
    """
    if isinstance(velocity, firedrake.Function):

        def v(x):
            return velocity.at(x, dont_raise=True)

    vx = v(initial_point)
    if vx is None:
        raise ValueError("Initial point is not inside the domain!")

    xs = [np.array(initial_point)]
    n = 0
    while n < max_num_points:
        n += 1
        speed = np.sqrt(sum(vx**2))
        x = xs[-1] + resolution / speed * vx
        vx = v(x)
        if vx is None:
            break
        xs.append(x)

    vy = v(initial_point)
    ys = [np.array(initial_point)]
    n = 0
    while n < max_num_points:
        n += 1
        speed = np.sqrt(sum(vy**2))
        y = ys[-1] - resolution / speed * vy
        vy = v(y)
        if vy is None:
            break
        ys.append(y)

    ys = ys[1:]

    return np.array(ys[::-1] + xs)


def _mesh_hmin(coordinates):
    cells = coordinates.cell_node_map().values
    vertices = coordinates.dat.data_ro

    hmin = np.inf
    _, vertices_per_cell = cells.shape
    for cell in cells:
        for n in range(vertices_per_cell):
            x = vertices[cell[n], :]
            for m in range(n + 1, vertices_per_cell):
                y = vertices[cell[m], :]
                hmin = min(hmin, sum((x - y) ** 2))

    return np.sqrt(hmin)


class StreamplotSet(mpl_streamplot.StreamplotSet, mpl_cmaps.ScalarMappable):
    def __init__(self, lines=None, arrows=None, norm=None, cmap=None):
        mpl_streamplot.StreamplotSet.__init__(self, lines, arrows)
        mpl_cmaps.ScalarMappable.__init__(self, norm=norm, cmap=cmap)
        self.set_array([])


def streamplot(u, **kwargs):
    r"""Draw streamlines of a vector field"""
    if u.ufl_shape != (2,):
        raise ValueError("Stream plots only defined for 2D vector fields!")

    u = _project_to_2d(u)
    axes = kwargs.pop("axes", plt.gca())
    cmap = kwargs.pop("cmap", mpl_cmaps.viridis)

    mesh = u.ufl_domain()
    coordinates = _get_coordinates(mesh)
    precision = kwargs.pop("precision", _mesh_hmin(coordinates))
    density = kwargs.pop("density", 2 * _mesh_hmin(coordinates))
    max_num_points = kwargs.pop("max_num_points", np.inf)
    coords = coordinates.dat.data_ro
    max_speed = icepack.norm(u, norm_type="Linfty")

    tree = scipy.spatial.KDTree(coords)
    indices = set(range(len(coords)))

    vmin = kwargs.get("vmin", 0)
    vmax = kwargs.get("vmax", max_speed)
    trajectories = []
    line_colors = []
    while indices:
        x0 = coords[indices.pop(), :]
        try:
            s = streamline(u, x0, precision, max_num_points)
            for y in s:
                for index in tree.query_ball_point(y, density):
                    indices.discard(index)

            points = s.reshape(-1, 1, 2)
            trajectories.extend(np.hstack([points[:-1], points[1:]]))

            speeds = np.sqrt(np.sum(np.asarray(u.at(s, tolerance=1e-10)) ** 2, 1))
            colors = (speeds - vmin) / (vmax - vmin)
            line_colors.extend(cmap(colors[:-1]))

        except ValueError:
            pass

    line_collection = LineCollection(trajectories, colors=np.array(line_colors))
    axes.add_collection(line_collection)
    axes.autoscale_view()

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return StreamplotSet(lines=line_collection, cmap=cmap, norm=norm)
