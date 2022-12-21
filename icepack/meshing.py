# Copyright (C) 2019-2020 by Daniel Shapero <shapero@uw.edu>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

r"""Utilities for turning glacier outlines into unstructured meshes"""

import copy
import itertools
import numpy as np
from numpy import linalg
import shapely.geometry
import geojson
import meshpy.triangle as triangle
import pygmsh
import firedrake

try:
    from firedrake.cython import dmcommon
except ImportError:
    from firedrake.cython import dmplex as dmcommon


def _flatten(features):
    r"""Expand all MultiLineString features in the input list to individual
    LineString features"""
    flat_features = []
    for feature in features:
        if feature["geometry"]["type"] == "LineString":
            flat_features.append(feature)
        if feature["geometry"]["type"] == "MultiLineString":
            properties = feature["properties"]
            for line_string in feature["geometry"]["coordinates"]:
                geometry = geojson.LineString(coordinates=line_string)
                flat_feature = geojson.Feature(geometry=geometry, properties=properties)
                flat_features.append(flat_feature)

    return flat_features


def _dist(x, y):
    return np.sqrt(sum((x - y) ** 2))


def _closest_endpoint(features, feature_index, point_index):
    r"""Return the feature and endpoint in a collection that is closest to the
    given feature and endpoint

    The result could be the opposite endpoint of the same feature.
    """
    feature = features[feature_index]
    x = np.array(feature["geometry"]["coordinates"][point_index])

    min_distance = np.inf
    min_findex = None
    min_pindex = None

    for findex in set(range(len(features))) - set([feature_index]):
        for pindex in (0, -1):
            y = features[findex]["geometry"]["coordinates"][pindex]
            distance = _dist(x, y)
            if distance < min_distance:
                min_distance = distance
                min_findex = findex
                min_pindex = pindex

    pindex = 0 if point_index == -1 else -1
    y = features[feature_index]["geometry"]["coordinates"][pindex]
    if _dist(x, y) < min_distance:
        min_findex = feature_index
        min_pindex = pindex

    return min_findex, min_pindex


def _compute_feature_adjacency(features):
    r"""Return a dictionary representing the adjacency between features

    For all feature indices `i` and endpoint indices `ei` in `(0, -1)`,
    `A[(i, ei)] = (j, ej)` where `j`, `ej` are the feature and endpoint
    indices of the adjacent segment.
    """
    adjacency = {}
    for i in range(len(features)):
        adjacency[(i, 0)] = _closest_endpoint(features, i, 0)
        adjacency[(i, -1)] = _closest_endpoint(features, i, -1)

    return adjacency


def _snap(input_features):
    r"""Reposition the endpoints of all features so that they are identical to
    the endpoint of the feature they are adjacent to"""
    features = copy.deepcopy(input_features)
    adjacency = _compute_feature_adjacency(features)

    for i in range(len(features)):
        for ei in (0, -1):
            j, ej = adjacency[(i, ei)]

            xi = features[i]["geometry"]["coordinates"][ei]
            xj = features[j]["geometry"]["coordinates"][ej]
            average = ((np.array(xi) + np.array(xj)) / 2).tolist()

            features[i]["geometry"]["coordinates"][ei] = average
            features[j]["geometry"]["coordinates"][ej] = average

    return features


def _powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def _reorient(input_features):
    r"""Flip the order of all features so that they are always oriented head-
    to-tail"""
    features = copy.deepcopy(input_features)
    n = len(features)
    adjacency = _compute_feature_adjacency(features)

    sets = _powerset(range(n))
    for s in sets:
        A = copy.deepcopy(adjacency)

        for i in s:
            j0, ej0 = A[(i, 0)]
            j1, ej1 = A[(i, -1)]

            A[(i, 0)] = (j1, ej1)
            A[(i, -1)] = (j0, ej0)

            A[(j0, ej0)] = (i, -1)
            A[(j1, ej1)] = (i, 0)

        if all([A[(i, ei)][1] != ei for i in range(n) for ei in (0, -1)]):
            for i in s:
                coords = features[i]["geometry"]["coordinates"]
                features[i]["geometry"]["coordinates"] = coords[::-1]

            return features

    raise ValueError("Input collection is not orientable! How did you even?!")


def _features_to_loops(features):
    r"""Return a list of lists of the segments in each loop, in order"""
    adjacency = _compute_feature_adjacency(features)

    # Add all the single-feature loops
    feature_indices = set(range(len(features)))
    loops = []
    for index in range(len(features)):
        if adjacency[(index, -1)] == (index, 0):
            loops.append([index])
            feature_indices.remove(index)

    # Add all the multi-feature loops
    while feature_indices:
        start_index = feature_indices.pop()
        loop = [start_index]
        index = adjacency[(start_index, -1)][0]
        while index != start_index:
            feature_indices.remove(index)
            loop.append(index)
            index = adjacency[(index, -1)][0]

        loops.append(loop)

    return loops


def _topologize(input_features):
    r"""Return a FeatureCollection of MultiLineStrings, one for each loop"""
    loops = _features_to_loops(input_features)
    features = []
    for loop in loops:
        coords = [list(geojson.utils.coords(input_features[index])) for index in loop]
        multi_line_string = geojson.MultiLineString(coords)
        features.append(geojson.Feature(geometry=multi_line_string))

    return features


def _find_bounding_feature(features):
    r"""Return the index of the feature in the collection that contains all
    other features"""
    line_strings = [sum(feature["geometry"]["coordinates"], []) for feature in features]
    polygons = [shapely.geometry.Polygon(line_string) for line_string in line_strings]

    for index, poly in enumerate(polygons):
        if all([poly.contains(p) for p in polygons if p is not poly]):
            return index

    raise ValueError("No polygon contains all other polygons!")


def _reorder(input_features):
    features = copy.deepcopy(input_features)
    index = _find_bounding_feature(features)
    bounding_feature = features.pop(index)
    return [bounding_feature] + features


def normalize(input_collection):
    r"""Normalize a GeoJSON feature collection into a form that can easily be
    transformed into the input for a mesh generator"""
    collection = copy.deepcopy(input_collection)
    for function in [_flatten, _snap, _reorient, _topologize, _reorder]:
        collection["features"] = function(collection["features"])

    return collection


def _add_loop_to_geometry(geometry, multi_line_string):
    line_loop = []
    for line_index, line_string in enumerate(multi_line_string):
        arc = []
        for index in range(len(line_string) - 1):
            x1 = line_string[index]
            x2 = line_string[index + 1]
            arc.append(geometry.add_line(x1, x2))

        num_lines = len(multi_line_string)
        next_line_string = multi_line_string[(line_index + 1) % num_lines]
        x1 = line_string[-1]
        x2 = next_line_string[0]
        arc.append(geometry.add_line(x1, x2))

        geometry.add_physical(arc)
        line_loop.extend(arc)

    return geometry.add_line_loop(line_loop)


def collection_to_geo(collection, lcar=10e3):
    r"""Convert a GeoJSON FeatureCollection into pygmsh geometry that can then
    be transformed into an unstructured triangular mesh"""
    collection = normalize(collection)
    features = collection["features"]

    geometry = pygmsh.built_in.Geometry()
    points = [
        [
            [
                geometry.add_point((point[0], point[1], 0.0), lcar=lcar)
                for point in line_string[:-1]
            ]
            for line_string in feature["geometry"]["coordinates"]
        ]
        for feature in features
    ]

    line_loops = [
        _add_loop_to_geometry(geometry, multi_line_string)
        for multi_line_string in points
    ]
    plane_surface = geometry.add_plane_surface(line_loops[0], line_loops[1:])
    geometry.add_physical(plane_surface)

    return geometry


def _find_interior_point(points):
    r"""Find a point inside the polygon described by the input points

    The input points don't need to be convex, but they do need to be ordered
    counter-clockwise"""
    # Find a point `X_0` outside the polygon with an x-coordinate distinct
    # from all of the polygon points
    xs = np.sort(points[:, 0])
    delta = np.diff(xs)
    index = np.argmax(delta)
    x = (xs[index] + xs[index + 1]) / 2
    ymin = points[:, 1].min()
    ymax = points[:, 1].max()
    y = ymax + (ymax - ymin) / 20
    X_0 = np.array([x, y])

    # Find all the points where the ray `X_0 + t * (0, -1)` intersects with
    # the input polygon
    intersections = []
    for index in range(len(points) - 1):
        X_1 = points[index, :]
        X_2 = points[index + 1, :]

        A = np.vstack((X_2 - X_1, (0, 1))).T
        b = X_1 - X_0
        try:
            s = linalg.solve(A, b)[0]
            if 0 <= s <= 1:
                intersections.append((1 - s) * X_1 + s * X_2)
        except linalg.LinAlgError:
            pass

    # Sort the points by distance from `X_0`; the average of the first two
    # points is inside the polygon.
    intersections = sorted(intersections, key=lambda X: np.sum((X - X_0) ** 2))
    return (intersections[0] + intersections[1]) / 2


def collection_to_triangle(collection, max_volume=None):
    r"""Convert a GeoJSON FeatureCollection into a Triangle geometry that can
    then be transformed into an unstructured triangular mesh"""
    collection = normalize(collection)

    coords = []
    edges = []
    markers = []
    num_edges = 0
    num_segments = 1
    for feature in collection["features"]:
        feature_coords = sum([l[:-1] for l in feature["geometry"]["coordinates"]], [])

        n = len(feature_coords)
        feature_edges = [(num_edges + i, num_edges + (i + 1) % n) for i in range(n)]
        num_edges += n

        feature_markers = []
        for l in feature["geometry"]["coordinates"]:
            feature_markers.extend([num_segments for i in range(len(l) - 1)])
            num_segments += 1

        coords.extend(feature_coords)
        edges.extend(feature_edges)
        markers.extend(feature_markers)

    holes = []
    for feature in collection["features"][1:]:
        feature_coords = sum([l[:-1] for l in feature["geometry"]["coordinates"]], [])
        point = _find_interior_point(np.array(feature_coords))
        holes.append(point)

    info = triangle.MeshInfo()
    info.set_points(coords)
    info.set_holes(holes)
    info.set_facets(edges, facet_markers=markers)
    return triangle.build(info, max_volume=max_volume)


def triangle_to_firedrake(mesh, comm=firedrake.COMM_WORLD):
    r"""Convert a generated Triangle geometry into a Firedrake mesh"""
    elements, points = mesh.elements, mesh.points
    # TODO: Remove this when we fully switch to the new Firedrake meshing
    # interface, `_from_cell_list` is deprecated
    if hasattr(firedrake.mesh, "plex_from_cell_list"):
        plex = firedrake.mesh.plex_from_cell_list(2, elements, points, comm)
    else:
        plex = firedrake.mesh._from_cell_list(2, elements, points, comm)

    markers = {
        tuple(sorted((v1, v2))): mesh.facet_markers[index]
        for index, (v1, v2) in enumerate(mesh.facets)
    }
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
    offset = plex.getDepthStratum(0)[0]
    for face in boundary_faces:
        vertices = tuple(sorted([v - offset for v in plex.getCone(face)]))
        marker = markers[vertices]
        plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, marker)

    return firedrake.Mesh(plex, reorder=False)
