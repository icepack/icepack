
import numpy as np

# -----------------
def read(file_stem):
    """
    Read in a mesh in Triangle's format.

    Parameters
    ==========
    file_stem: the stem of the filename for the mesh, so if the mesh is stored
      in the files <meshname>.X.node, <meshname>.X.ele, etc. then the argument
      should be the string "<meshname>.X"

    Returns:
    =======
    x, y: coordinates of the mesh points
    triangles: (num_triangles, 3)-array of nodes in each triangle
    bnd: boundary marker of each node
    """

    # TODO: check which mesh files are actually found and determine whether
    # we're reading a mesh or a PSLG

    x, y, bnd = _read_node(file_stem)
    triangles = _read_ele(file_stem)

    return x, y, triangles, bnd



# ------------------
def write(file_stem, x = None, y = None, bnd = None, triangles = None):
    """
    Write out a triangular mesh in Triangle's format.

    Paramters:
    =========
    file_stem: the start of the mesh filename; writes the mesh nodes to
      file_stem.node, the elements to file_stem.ele, etc.
    x, y: the node positions
    """

    # TODO: raise a type error if keyword arguments aren't supplied

    num_nodes = len(x)
    with open(file_stem + ".node", "w") as node_file:
        node_file.write("{0} 2 0 1\n".format(num_nodes))
        for i in range(num_nodes):
            node_file.write("{0} {1} {2} {3}\n".format(i+1, x[i], y[i], bnd[i]))

    num_triangles, _ = np.shape(triangles)
    with open(file_stem + ".ele", "w") as ele_file:
        ele_file.write("{0} 3 0\n".format(num_triangles))
        for i in range(num_triangles):
            t = [k+1 for k in triangles[i]]
            ele_file.write("{0} {1} {2} {3}\n".format(i+1, t[0], t[1], t[2]))

    return


# ------------------------------------------------
def write_poly(filename, x, y, bnd, edges, xh, yh):
    """
    Write a planar straight-line graph to a .poly file
    """
    with open(filename, "w") as poly_file:
        num_nodes = len(x)
        poly_file.write("{0} 2 0 1\n".format(num_nodes))

        for k in range(num_nodes):
            poly_file.write("{0} {1} {2} {3}\n"
                            .format(k+1, x[k], y[k], bnd[k]))

        poly_file.write("{0} 1\n".format(num_nodes))

        num_edges, _ = np.shape(edges)
        for n in range(num_edges):
            i, j = edges[n, :]
            poly_file.write("{0} {1} {2} {3}\n"
                            .format(n+1, i+1, j+1, max(bnd[i], bnd[j])))

        num_holes = len(xh)
        poly_file.write("{0}\n".format(num_holes))
        for n in range(num_holes):
            poly_file.write("{0} {1} {2}\n".format(n+1, xh[n], yh[n]))

    return


# TODO: reading mesh files should skip escaped lines that start with a #

# -----------------------
def _read_node(file_stem):
    with open(file_stem + ".node", "r") as node_file:
        num_nodes = int(node_file.readline().split()[0])
        x = np.zeros(num_nodes, dtype = np.float64)
        y = np.zeros(num_nodes, dtype = np.float64)
        bnd = np.zeros(num_nodes, dtype = np.int32)

        for i in range(num_nodes):
            line = node_file.readline().split()
            x[i], y[i] = float(line[1]), float(line[2])
            bnd[i] = int(line[3])

    return x, y, bnd


# ----------------------
def _read_ele(file_stem):
    with open(file_stem + ".ele", "r") as ele_file:
        num_triangles = int(ele_file.readline().split()[0])
        triangles = np.zeros((num_triangles, 3), dtype = np.int32)

        for i in range(num_triangles):
            triangles[i, :] = [int(k) - 1 for k in
                               ele_file.readline().split()[1:]]

    return triangles
