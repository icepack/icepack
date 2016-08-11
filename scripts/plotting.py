
import numpy as np
import matplotlib.pyplot as plt
import sys
import ucd

def _plot_field(x, y, q, png_filename):
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, q, 36, shading='faceted', cmap = 'viridis')
    plt.colorbar()
    plt.savefig(png_filename)
    return

if __name__ == "__main__":
    try:
        x, y, _, q = ucd.read(sys.argv[1])
    except ValueError:
        x, y, _, u, v = ucd.read(sys.argv[1])
        q = np.sqrt(u**2 + v**2)

    _plot_field(x, y, q, sys.argv[2])
