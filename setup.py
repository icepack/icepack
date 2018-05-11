# Copyright (C) 2017-2018 by Daniel Shapero <shapero@uw.edu>
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

from glob import glob
from os.path import basename, splitext
from setuptools import setup, find_packages

setup(
    name='icepack',
    version='0.0.3',
    license='GPL v3',
    description='ice sheet flow modelling with the finite element method',
    author='Daniel Shapero',
    url='https://github.com/icepack/icepack',
    packages=find_packages(exclude=['doc', 'test']),
    install_requires=['firedrake', 'numpy', 'scipy', 'matplotlib', 'rasterio']
)
