
import numpy as np

from icepack.mesh import QuadMesh

# -------------------------------------------
def write_tri(filename, x, y, bnd, triangles):
    """
    Write a triangular mesh to the gmsh .msh format.
    """
    num_nodes = len(x)
    num_triangles, _ = np.shape(triangles)

    with open(filename, "w") as msh_file:
        msh_file.write("$MeshFormat\n")
        msh_file.write("2.0 0 8\n")
        msh_file.write("$EndMeshformat\n")

        msh_file.write("$Nodes\n")
        msh_file.write("{0}\n".format(num_nodes))
        for i in range(num_nodes):
            msh_file.write("{0} {1} {2} 0.0\n".format(i+1, x[i], y[i]))
        msh_file.write("$EndNodes\n")

        msh_file.write("$Elements\n")
        msh_file.write("{0}\n".format(num_triangles))
        for n in range(num_triangles):
            t = [k + 1 for k in triangles[n, :]]
            msh_file.write("{0} 2 0 {1} {2} {3}\n".format(n+1, t[0], t[1], t[2]))
        msh_file.write("$Endelements\n")

    return


# ---------------------
def read_quad(filename):
    """
    Read an unstructured quad mesh in the gmsh .msh format.
    """
    with open(filename, "r") as msh_file:
        for k in range(4):
            msh_file.readline()

        num_nodes = int(msh_file.readline())
        x, y = np.zeros(num_nodes), np.zeros(num_nodes)

        for k in range(num_nodes):
            line = msh_file.readline().split()
            x[k], y[k] = float(line[1]), float(line[2])

        for k in range(2):
            msh_file.readline()

        num_elements = int(msh_file.readline())
        quads = []
        for k in range(num_elements):
            line = [int(s) for s in msh_file.readline().split()]
            element_type = line[1]

            if element_type == 3:
                num_tags = line[2]
                start_index = 3 + num_tags
                quad = [n - 1 for n in line[start_index: start_index + 4]]
                quads.append(quad)

        quads = np.array(quads)

    return QuadMesh(x, y, quads)
