
import numpy as np

N = 256
Pi = 4.0 * np.arctan(1.0)
cl = 1.0e22

fid = open("circle.geo", "w")

fid.write("// Mesh points\n")
for k in range(N):
    fid.write("Point({0}) = {{{1}, {2}, 0.0, {3}}};\n"
              .format(k + 1, np.cos(2*Pi*k/N), np.sin(2*Pi*k/N), cl))
fid.write("\n")

fid.write("// Mesh edges\n")
for k in range(N):
    fid.write("Line({0}) = {{{1}, {2}}};\n"
              .format(k + 1, k + 1, 1 + (k + 1) % N))
fid.write("\n")

fid.write("// Line loop for mesh boundary\n")
fid.write("Line Loop({0}) = {{{1}"
          .format(N + 1, 1))
for k in range(1, N):
    fid.write(", {0}".format(k + 1))
fid.write("};\n\n")

fid.write("Plane surface({0}) = {{{1}}};\n".format(N + 2, N + 1))

fid.write("Mesh.Algorithm = 8;\n")
fid.write("Recombine Surface{{{0}}};\n".format(N + 2))
