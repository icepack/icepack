
import numpy as np
import matplotlib.pyplot as plt
import sys
import ucd

def velocity(ucd_filename, png_filename):
    x, y, _, u, v = ucd.read(ucd_filename + ".ucd")
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(x, y, np.sqrt(u**2 + v**2), 36, shading='faceted')
    plt.colorbar()
    plt.savefig(png_filename + ".png")
    return

if __name__ == "__main__":
    velocity(sys.argv[1], sys.argv[2])
