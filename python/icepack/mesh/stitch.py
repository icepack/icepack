
import copy
import itertools
from numpy import ones, zeros, sqrt
from matplotlib.path import *


# --------------
def dist(x1, x2):
    return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)


# -----------------------------------
class IsolatedSegmentError(Exception):
    pass


# -----------------------------------
def next_segment(Xs, i, tol = 1.0e4):
    """
    Return the next segment in the input geometry

    Args:
        Xs: a list of coordinates of the lines of the input geometry
        i: an index of one line of the input geometry
        tol: criterion for whether another segment's endpoint is close enough

    Returns:
        j: the index of the segment after `i` in the input geometry, i.e. the
            segment whose start- or end-point is closest to the end-point of
            `i`. This could be `i` itself if it describes a closed loop. If the
            successor segment's order needs to be reversed, returns `-j`.

    Raises:
        IsolatedSegmentError on a segment with no successor within the given
            tolerance.
    """
    num_segments = len(Xs)

    Xi = Xs[i]
    if dist(Xi[0], Xi[-1]) < tol:
        return i

    for j in range(num_segments):
        if j != i:
            Xj = Xs[j]

            if dist(Xi[-1], Xj[0]) < tol:
                return j

            if dist(Xi[-1], Xj[-1]) < tol:
                return -j

    raise IsolatedSegmentError()


# --------------------------------------
def segment_successors(Xs, tol = 1.0e4):
    """
    Return a new geometry identical to the input but with orientations flipped
    so that all segments lie end-to-end.

    Args:
        Xs: input geometry
        tol: tolerance for segment proximity

    Returns:
        Ys: input geometry, possibly with some segments in reverse order
        successors: successors[i] = the next segment after `i` in the PSLG
    """
    num_segments = len(Xs)

    Ys = copy.deepcopy(Xs)

    segments = set(range(num_segments))
    successors = list(range(num_segments))

    while segments:
        i0 = segments.pop()

        i = i0
        j = next_segment(Ys, i, tol)
        while j != i0:
            if j < 0:
                j = -j
                Ys[j].reverse()

            segments.remove(j)
            successors[i] = j

            i = j
            j = next_segment(Ys, i, tol)

        successors[i] = i0

    return Ys, successors


# --------------------------------
def lines_to_paths(Xs, successors):
    """
    Return a list of closed matplotlib Path objects of the input geometry
    """
    segments = set(range(len(Xs)))
    Ps = []

    while segments:
        i0 = segments.pop()
        i = i0

        X = X[i]

        j = successors[i]
        while j != i0:
            segments.remove(j)
            X.extend(Xs[j])

            i = j
            j = successors[i]

        p = Path(X, closed = True)
        Ps.append(p)

    return Ps


# ---------------------------
def find_point_inside_path(p):
    """
    Return a point inside the path p.
    Triangle needs to have a point contained in any holes in the mesh.
    """

    x = (0.0, 0.0)

    i, j = 0, len(p)/2
    while not p.contains_point(x):
        j += 1
        x = (0.5 * p.vertices[i, 0] + p.vertices[j, 0],
             0.5 * p.vertices[i, 1] + p.vertices[j, 1])

    return x


# --------------------------------
def identify_holes(Xs, successors):
    """
    Return a list of points
    """

    Ys = []
    Ps = lines_to_paths(Xs, successors)
    for p, q in itertools.combinations(ps, 2):
        if p.contains_path(q):
            y = point_inside_path(q)
            Ys.append(y)

    return Ys
