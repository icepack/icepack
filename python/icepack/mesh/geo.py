
import numpy as np


# -------------------------------------------------------------
def write(filename, Xs, successors, dx = 1.0e+22, quad = False):
    """
    Write out a PSLG to the gmsh .geo format
    """
    num_segments = len(Xs)

    with open(filename, "w") as geo_file:
        geo_file.write("cl = {0};\n".format(dx))

        # Write out the PSLG points
        point_count = 1
        for X in Xs:
            for x in X:
                geo_file.write("Point({0}) = {{{1}, {2}, 0.0, cl}};\n"
                               .format(point_count, x[0], x[1]))
                point_count += 1
        geo_file.write("\n")

        # indexing of each edge into the list of all edges
        offsets = np.zeros(num_segments + 1, dtype = int)
        offsets[0] = 1
        for k in range(num_segments):
            offsets[k + 1] = offsets[k] + len(Xs[k])

        # Write out the PSLG edges
        edge_count = point_count
        segments = set(range(num_segments))
        line_loops = []
        while segments:
            # Pick some segment that we haven't looked at yet
            k0 = segments.pop()
            k = k0

            loop = []
            while True:
                segment_length = len(Xs[k])

                # Write out all the edges for the current segment
                for i in range(segment_length - 1):
                    geo_file.write("Line({0}) = {{{1}, {2}}};\n"
                                   .format(edge_count + i,
                                           offsets[k] + i,
                                           offsets[k] + i + 1))
                    loop.append(edge_count + i)

                # Write out the edge connecting the end of the current segment
                # to the beginning of the next segment
                l = successors[k]
                geo_file.write("Line({0}) = {{{1}, {2}}};\n"
                               .format(edge_count + segment_length - 1,
                                       offsets[k] + segment_length - 1,
                                       offsets[l]))

                loop.append(edge_count + segment_length - 1)
                edge_count += segment_length

                # If the next segment is the first one, we're done with this
                # connected component
                if l == k0:
                    break

                segments.remove(l)
                k = l

            line_loops.append(loop)
        geo_file.write("\n")

        # Write out the line loops of the PSLG
        line_loop_count = edge_count
        plane_surface = []
        for loop in line_loops:
            geo_file.write("Line Loop({0}) = {{{1}}};\n"
                           .format(line_loop_count,
                                   ', '.join([str(k) for k in loop])))
            plane_surface.append(line_loop_count)
            line_loop_count += 1
        geo_file.write("\n")

        # Write out a plane surface for the whole PSLG
        geo_file.write("Plane Surface({0}) = {{{1}}};\n\n"
                       .format(line_loop_count,
                               ', '.join([str(k) for k in plane_surface])))

        if quad:
            geo_file.write("Recombine Surface{{{0}}};\n"
                           .format(line_loop_count))
            geo_file.write("Mesh.SubdivisionAlgorithm=1;\n")
            geo_file.write("Mesh.Algorithm=8;\n")
