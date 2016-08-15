
import numpy as np
import matplotlib.pyplot as plt

from icepack import ucd

if __name__ == "__main__":
    x, y, cells, u, v = ucd.read("v.ucd")
    _, _, _, uo, vo = ucd.read("vo.ucd")

    triangles = ucd.quad_cells_to_triangles(x, y, cells)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal')
    ctr = ax.tricontourf(x/1000.0, y/1000.0, triangles,
                         np.sqrt((u - uo)**2 + (v - vo)**2),
                         cmap = 'viridis', shading = 'faceted')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    cbar = fig.colorbar(ctr)

    plt.show()
