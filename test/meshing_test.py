# Copyright (C) 2019 by Daniel Shapero <shapero@uw.edu>
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

import pytest
import numpy as np
from numpy import pi as π
import geojson
import icepack.meshing


def needs_snapping():
    coords = [[(0., -1e-6), (1., 0.), (1., 1. - 1e-6)],
              [(1., 1. + 1e-6), (0., 1.), (0., 1e-6)]]
    multi_line_string = geojson.MultiLineString(coords, validate=True)
    feature = geojson.Feature(geometry=multi_line_string, properties={})
    return geojson.FeatureCollection([feature])


def needs_reorienting():
    coords = [[(0., 0.), (1., 0.), (1., 1.)],
              [(0., 0.), (0., 1.), (1., 1.)]]
    multi_line_string = geojson.MultiLineString(coords, validate=True)
    feature = geojson.Feature(geometry=multi_line_string, properties={})
    return geojson.FeatureCollection([feature])


def has_interior():
    coords = [(0., 0.), (1., 0.), (1., 1.), (0., 1.), (0., 0.)]
    outer_line_string = geojson.LineString(coords, validate=True)

    r = 1 / 8
    coords = [(0.5 + r * np.cos(θ), 0.5 + r * np.sin(θ))
              for θ in np.linspace(0, 2 * π, 256)]
    inner_line_string = geojson.LineString(coords, validate=True)

    outer_feature = geojson.Feature(geometry=outer_line_string, properties={})
    inner_feature = geojson.Feature(geometry=inner_line_string, properties={})
    return geojson.FeatureCollection([outer_feature, inner_feature])


test_data = [
    needs_snapping,
    needs_reorienting,
    has_interior
]


@pytest.mark.parametrize('input_data', test_data)
def test_normalize(input_data):
    collection = input_data()
    result = icepack.meshing.normalize(collection)
    assert result == icepack.meshing.normalize(result)


@pytest.mark.parametrize('input_data', test_data)
def test_converting_to_geo(tmpdir, input_data):
    collection = input_data()
    geometry = icepack.meshing.collection_to_geo(collection, lcar=1e-2)
    assert geometry.get_code() is not None
