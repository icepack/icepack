
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
from matplotlib.collections import LineCollection
import numpy as np
import firedrake

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

    figure = plt.figure()
    if axes is None:
        axes = figure.add_subplot(111, **kwargs)

    # Add lines for all of the edges, internal or boundary
    cell = coordinates.cell_node_map().values
    vertices = coords[cell[:, (0, 1, 2, 0)]]
    lines = LineCollection(vertices, colors=(0.0, 0.0, 0.0))
    axes.add_collection(lines)

    # Add colored lines for the boundary edges
    facets = mesh.topology.exterior_facets
    local_facet_id = facets.local_facet_dat.data_ro
    markers = facets.unique_markers
    clrs = _get_colors(colors, len(markers))

    for marker in markers:
        indices = facets.subset(int(marker)).indices
        n = len(indices)
        roll = 2 - local_facet_id[indices]
        cell = coordinates.exterior_facet_node_map().values[indices, :]
        edges = np.array([np.roll(cell[k,:], roll[k]) for k in range(n)])[:,:2]
        vertices = coords[edges]
        lines = LineCollection(vertices, color=clrs[marker-1], label=marker)
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

    return axes


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

    return firedrake.plot(mesh_or_function, axes=axes, **kwargs)

