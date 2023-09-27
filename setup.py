# Copyright (C) 2017-2021 by Daniel Shapero <shapero@uw.edu>
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

from setuptools import setup, find_packages

setup(
    name="icepack",
    version="1.0.1",
    license="GPL v3",
    description="ice sheet flow modelling with the finite element method",
    author="Daniel Shapero",
    url="https://github.com/icepack/icepack",
    packages=find_packages(exclude=["doc", "test"]),
    package_data={"icepack": ["registry-nsidc.txt", "registry-outlines.txt"]},
    install_requires=[
        "numpy",
        "scipy<=1.9.3",
        "matplotlib",
        "rasterio>=1.2.7",
        "netCDF4",
        "xarray",
        "geojson",
        "shapely",
        "pooch>=1.0.0",
        "gmsh",
        "pygmsh<=6.1.1",
        "meshio>=3.3.1",
        "MeshPy",
        "tqdm",
    ],
    extras_require={
        "doc": ["sphinx", "ipykernel", "nbconvert", "nikola"],
        "opt": [
            "roltrilinos @ git+https://github.com/icepack/Trilinos.git",
            "ROL @ git+https://github.com/icepack/pyrol.git",
        ],
    },
)
