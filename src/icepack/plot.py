# Copyright (C) 2017-2024 by Daniel Shapero <shapero@uw.edu>
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

r"""Plotting utilities"""

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def subplots(*args, **kwargs):
    r"""Make some subplots but make all the aspect ratios equal and always use
    scientific notation for the axis tick labels"""
    subplot_kw = kwargs.get("subplot_kw", {})
    subplot_kw["adjustable"] = subplot_kw.get("adjustable", "box")
    kwargs["subplot_kw"] = subplot_kw
    fig, axes = plt.subplots(*args, **kwargs)

    def fmt(ax):
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=True))
        ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

    try:
        if len(axes.shape) == 1:
            for ax in axes:
                fmt(ax)
        else:
            for row in axes:
                for ax in row:
                    fmt(ax)
    except AttributeError:
        fmt(axes)

    return fig, axes
