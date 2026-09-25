# Copyright (C) 2017-2026 by Daniel Shapero <shapero@uw.edu>
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

from icepack.norms import norm
from icepack.interpolate import interpolate, fit
from icepack.utilities import compute_surface
from icepack.calculus import vertical_velocity, depth_average, lift3d
from icepack.models.viscosity import rate_factor
import icepack.meshing
import icepack.datasets
import icepack.models
import icepack.solvers
import icepack.statistics

__all__ = [
    "norm",
    "interpolate",
    "fit",
    "depth_average",
    "lift3d",
    "compute_surface",
    "rate_factor",
    "meshing",
    "datasets",
    "models",
    "solvers",
    "statistics",
]
