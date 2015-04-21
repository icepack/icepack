
import numpy as np

N = 32
Pi = 4.0 * np.arctan(1.0)
cl = 1.0e22

fid = open("half_circle.geo", "w")

fid.write("// Mesh points\n")
for k in range(N + 1):
    fid.write("Point({0}) = {{{1}, {2}, 0.0, {3}}};\n"
              .format(k + 1, 5e3 * np.cos(Pi * k / N), 5e3 * np.sin(Pi * k / N), cl))
fid.write("\n")

fid.write("// Mesh edges\n")
for k in range(N + 1):
    fid.write("Line({0}) = {{{1}, {2}}};\n"
              .format(k + 1, k + 1, 1 + (k + 1) % (N + 1)))
fid.write("\n")

fid.write("// Line loop for mesh boundary\n")
fid.write("Line Loop({0}) = {{{1}"
          .format(N + 2, 1))
for k in range(1, N + 1):
    fid.write(", {0}".format(k + 1))
fid.write("};\n\n")

fid.write("Plane Surface({0}) = {{{1}}};\n".format(N + 3, N + 2))

# fid.write("Mesh.Algorithm = 8;\n")
fid.write("Recombine Surface{{{0}}};\n".format(N + 3))
