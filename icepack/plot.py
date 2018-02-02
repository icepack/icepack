
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
from matplotlib.collections import LineCollection
import numpy as np
import firedrake
from icepack.grid import GridData

def _get_coordinates(mesh):
    """Return the coordinates of a mesh if the mesh is piecewise linear,
    or interpolate them to piecewise linear if the mesh is curved
    """
    coordinates = mesh.coordinates
    element = coordinates.function_space().ufl_element()
    if element.degree() != 1:
        from firedrake import VectorFunctionSpace, interpolate
        V = VectorFunctionSpace(mesh, element.family(), 1)
        coordinates = interpolate(coordinates, V)

    return coordinates


def _get_colors(colors, num_markers):
    if colors is None:
        cmap = matplotlib.cm.get_cmap('Dark2')
        return cmap([k / num_markers for k in range(num_markers)])

    return matplotlib.colors.to_rgba_array(colors)


def plot_mesh(mesh, colors=None, axes=None, **kwargs):
    if (mesh.geometric_dimension() != 2) or (mesh.topological_dimension() != 2):
        raise NotImplementedError("Plotting meshes only implemented for 2D")

    if mesh.ufl_cell().cellname() == "quadrilateral":
        raise NotImplementedError("Plotting meshes only implemented for "
                                  "triangles")

    mesh.init() # Apprently this doesn't happen automatically?
    coordinates = _get_coordinates(mesh)
    coords = coordinates.dat.data_ro

    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111)

    # Add lines for all of the edges, internal or boundary
    cell = coordinates.cell_node_map().values
    vertices = coords[cell[:, (0, 1, 2, 0)]]
    lines = LineCollection(vertices, colors=(0.0, 0.0, 0.0), **kwargs)
    axes.add_collection(lines)

    # Add colored lines for the boundary edges
    facets = mesh.topology.exterior_facets
    local_facet_id = facets.local_facet_dat.data_ro
    markers = facets.unique_markers
    clrs = _get_colors(colors, len(markers))

    for i, marker in enumerate(markers):
        indices = facets.subset(int(marker)).indices
        n = len(indices)
        roll = 2 - local_facet_id[indices]
        cell = coordinates.exterior_facet_node_map().values[indices, :]
        edges = np.array([np.roll(cell[k,:], roll[k]) for k in range(n)])[:,:2]
        vertices = coords[edges]
        lines = LineCollection(vertices, color=clrs[i], label=marker,
                               **kwargs)
        axes.add_collection(lines)
    axes.legend()

    # Adjust the axis limits
    for setter, k in zip(["set_xlim", "set_ylim", "set_zlim"],
                         range(coords.shape[1])):
        amin, amax = coords[:, k].min(), coords[:, k].max()
        extra = (amax - amin) / 20
        amin -= extra
        amax += extra
        getattr(axes, setter)(amin, amax)
    axes.set_aspect("equal")

    axes.tick_params(axis='x', rotation=-30)
    return axes


def plot_grid_data(grid_data, axes=None, **kwargs):
    """Plot a gridded data object"""
    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111)

    ny, nx = grid_data.shape
    x0, x1 = grid_data.coordinate(0, 0), grid_data.coordinate(ny - 1, nx - 1)

    x = np.linspace(x0[0], x1[0], nx)
    y = np.linspace(x0[1], x1[1], ny)

    axes.contourf(x, y, grid_data.data, **kwargs)

    axes.tick_params(axis='x', rotation=-30)
    return axes


def streamline(velocity, initial_point, resolution,
               max_num_points=np.inf, backwards=False):
    """Return a streamline of a 2D velocity field

    A streamline :math:`\\gamma` of a velocity field :math:`v` is a curve
    that solves the ordinary differential equation

    .. math::
       \\frac{d\\gamma}{dt} = v(\\gamma)

    This function returns an approximate streamline for a velocity field.
    Streamlines are primarily for plotting vector fields but are useful for
    other kinds of postprocessing of vector field data.

    Parameters
    ----------
    velocity : firedrake.Function or tuple of icepack.grid.GridData
        the velocity field we are integrating
    initial_point : pair of floats
        the starting point for the streamline
    resolution : float
        the desired length of each segment of the streamline
    max_num_points : int
        maximum number of points of the streamline; can be necessary to set
        if the trajectory can spiral around a center node
    backwards : bool
        whether to integrate the streamline in the reverse direction
        (defaults to `False`)

    Returns
    -------
    xs : numpy array of points
    """
    if isinstance(velocity, firedrake.Function):
        def v(x):
            return velocity.at(x, dont_raise=True)
    else:
        def v(x):
            try:
                if velocity[0].is_masked(x) or velocity[1].is_masked(x):
                    return None
            except ValueError:
                return None

            return np.array((velocity[0](x), velocity[1](x)))

    sign = -1 if backwards else +1

    vx = v(initial_point)
    if vx is None:
        raise ValueError("Initial point is not inside the domain!")

    xs = [np.array(initial_point)]

    n = 0
    while n < max_num_points:
        n += 1
        speed = np.sqrt(sum(vx**2))
        x = xs[-1] + sign * resolution / speed * vx
        vx = v(x)
        if vx is None:
            break
        xs.append(x)

    return np.array(xs)


def plot(mesh_or_function, axes=None, **kwargs):
    """Make a visualization of a mesh or a field

    This function overrides the usual firedrake plotting function so that
    meshes are shown with colors and a legend for different parts of the
    boundary.

    .. seealso::

       :py:func:`firedrake.plot.plot`
          Documentation for the firedrake plot function
    """
    if isinstance(mesh_or_function, firedrake.mesh.MeshGeometry):
        return plot_mesh(mesh_or_function, axes=axes, **kwargs)

    if isinstance(mesh_or_function, GridData):
        return plot_grid_data(mesh_or_function, axes=axes, **kwargs)

    axes = firedrake.plot(mesh_or_function, axes=axes, **kwargs)
    axes.tick_params(axis='x', rotation=-30)
    return axes

