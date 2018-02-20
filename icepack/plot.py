
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
from matplotlib.collections import LineCollection
import numpy as np
import scipy.spatial
import firedrake
import icepack
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


def _fix_axis_limits(axes, coords):
    for setter, k in zip(["set_xlim", "set_ylim", "set_zlim"],
                         range(coords.shape[1])):
        amin, amax = coords[:, k].min(), coords[:, k].max()
        extra = (amax - amin) / 20
        amin -= extra
        amax += extra
        getattr(axes, setter)(amin, amax)

    axes.tick_params(axis='x', rotation=-30)
    axes.set_aspect("equal")
    return axes


def plot_mesh(mesh, colors=None, axes=None, **kwargs):
    """Plot a mesh with a different color for each boundary segment"""
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

    return _fix_axis_limits(axes, coords)


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


def streamline(velocity, initial_point, resolution, max_num_points=np.inf):
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
            x = vertices[cell[n],:]
            for m in range(n + 1, vertices_per_cell):
                y = vertices[cell[m],:]
                hmin = min(hmin, sum((x - y)**2))

    return np.sqrt(hmin)

def _plot_vector_field_streamline(v, axes=None, resolution=None, spacing=None,
                                  max_num_points=np.inf, **kwargs):
    mesh = v.ufl_domain()
    coordinates = _get_coordinates(mesh)
    if resolution is None:
        resolution = _mesh_hmin(coordinates)
    if spacing is None:
        spacing = 2 * resolution

    coords = coordinates.dat.data_ro
    tree = scipy.spatial.KDTree(coords)

    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111)

    cmap = kwargs.pop("cmap", matplotlib.cm.viridis)
    norm = icepack.norm(v, 'Linfty')

    indices = set(range(len(coords)))
    while len(indices) > 0:
        x = coords[indices.pop(), :]
        try:
            s = streamline(v, x, resolution, max_num_points=max_num_points)
            speeds = np.sqrt(np.sum(np.asarray(v.at(s, tolerance=1e-10))**2, 1))
            colors = speeds / norm
            segments = [(s[k, :], s[k+1,:]) for k in range(len(s) - 1)]
            lines = LineCollection(segments, colors=cmap(colors[:-1]))
            axes.add_collection(lines)

            for z in s:
                for index in tree.query_ball_point(z, spacing):
                    indices.discard(index)
        except ValueError:
            pass

    return _fix_axis_limits(axes, coords)


def _plot_vector_field_magnitude(v, axes=None, **kwargs):
    mesh = v.function_space().mesh()
    element = v.function_space().ufl_element()
    Q = firedrake.FunctionSpace(mesh, element.family(), element.degree())
    from firedrake import inner, sqrt
    magnitude = firedrake.Function(Q).interpolate(sqrt(inner(v, v)))

    return plot(magnitude, axes=axes, **kwargs)


def plot_vector_field(v, axes=None, **kwargs):
    """Plot the directions, streamlines, or magnitude of a vector field

    The default method to plot a vector field is to compute the magnitude
    at each point and make a contour plot of this scalar field. You can also
    make a quiver plot by passing the keyword argument `method='quiver'`.
    Finally, the streamlines of the vector field can be plotted with colors
    representing the magnitude by passing `method='streamline'`, although
    this is substantially more expensive and may require tweaking.

    Parameters
    ----------
    v : firedrake.Function
        The vector field to plot
    axes : matplotlib.Axes, optional
        The axis to draw the figure on

    Other Parameters
    ----------------
    method : str, optional
        Either 'magnitude' (default), 'streamline', or 'quiver'
    resolution : float, optional
        If using a streamline plot, the resolution along a streamline
    spacing : float, optional
        If using a streamline plot, the minimum spacing between seed points
        for streamlines
    max_num_points : int, optional
        If using a streamline plot, the maximum number of points along a
        streamline; this is probably necessary if the vector field has a
        stable equilibrium
    """
    method_name = kwargs.pop('method', 'magnitude')

    methods = {"magnitude": _plot_vector_field_magnitude,
               "streamline": _plot_vector_field_streamline,
               "quiver": firedrake.plot}

    if not method_name in methods:
        raise ValueError("Method for plotting vector field must be either "
                        "`magnitude`, `streamline`, or `quiver`!")

    return methods[method_name](v, axes=axes, **kwargs)


def plot(mesh_or_function, axes=None, **kwargs):
    """Make a visualization of a mesh or a field

    This function overrides the usual firedrake plotting functions. When
    plotting a mesh, the boundaries are shown with colors and a legend to
    show the ID for each boundary segment. When plotting a scalar or vector
    field, the default colormap is set to viridis. Finally, when plotting a
    vector field, the default behavior is to instead plot the magnitude of
    the vector, rather than a quiver plot. A quiver plot or a streamline
    plot can instead be selected by passing the keyword argument `method`
    with value `quiver` or `streamline` respectively.

    .. seealso::

       :py:func:`firedrake.plot.plot`
          Documentation for the firedrake plot function
    """
    if isinstance(mesh_or_function, firedrake.mesh.MeshGeometry):
        return plot_mesh(mesh_or_function, axes=axes, **kwargs)

    kwargs['cmap'] = matplotlib.cm.get_cmap(kwargs.pop('cmap', 'viridis'))

    if isinstance(mesh_or_function, GridData):
        return plot_grid_data(mesh_or_function, axes=axes, **kwargs)

    if (len(mesh_or_function.ufl_shape) == 1):
        return plot_vector_field(mesh_or_function, axes=axes, **kwargs)

    axes = firedrake.plot(mesh_or_function, axes=axes, **kwargs)
    axes.tick_params(axis='x', rotation=-30)
    return axes
