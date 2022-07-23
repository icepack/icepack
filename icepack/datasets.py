# Copyright (C) 2019-2021 by Daniel Shapero <shapero@uw.edu>
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

r"""Routines for fetching the glaciological data sets used in the demos"""

import os
from getpass import getpass
import pkg_resources
import requests
import pooch
import warnings


pooch.get_logger().setLevel("WARNING")


class EarthDataDownloader:
    def __init__(self):
        self._username = None
        self._password = None

    def _get_credentials(self):
        if self._username is None:
            username_env = os.environ.get("EARTHDATA_USERNAME")
            if username_env is None:
                self._username = input("EarthData username: ")
            else:
                self._username = username_env

        if self._password is None:
            password_env = os.environ.get("EARTHDATA_PASSWORD")
            if password_env is None:
                self._password = getpass("EarthData password: ")
            else:
                self._password = password_env

        return self._username, self._password

    def __call__(self, url, output_file, dataset):
        auth = self._get_credentials()
        downloader = pooch.HTTPDownloader(auth=auth, progressbar=True)
        try:
            login = requests.get(url)
            downloader(login.url, output_file, dataset)
        except requests.exceptions.HTTPError as error:
            if "Unauthorized" in str(error):
                pooch.get_logger().error("Wrong username/password!")
                self._username = None
                self._password = None
            raise error


_earthdata_downloader = EarthDataDownloader()


nsidc_data = pooch.create(path=pooch.os_cache("icepack"), base_url="", registry=None)

registry_nsidc = pkg_resources.resource_stream("icepack", "registry-nsidc.txt")
nsidc_data.load_registry(registry_nsidc)


def fetch_measures_antarctica():
    r"""Fetch the MEaSUREs Antarctic velocity map"""
    return nsidc_data.fetch(
        "antarctic_ice_vel_phase_map_v01.nc", downloader=_earthdata_downloader
    )


def fetch_measures_greenland():
    r"""Fetch the MEaSUREs Greenland velocity map"""
    return [
        nsidc_data.fetch(
            f"greenland_vel_mosaic200_2015_2016_{field_name}_v02.1.tif",
            downloader=_earthdata_downloader,
        )
        for field_name in ["vx", "vy", "ex", "ey"]
    ]


def fetch_bedmachine_antarctica():
    r"""Fetch the BedMachine map of Antarctic ice thickness, surface elevation,
    and bed elevation"""
    return nsidc_data.fetch(
        "BedMachineAntarctica_2020-07-15_v02.nc", downloader=_earthdata_downloader
    )


def fetch_bedmachine_greenland():
    r"""Fetch the BedMachine map of Greenland ice thickness, surface elevation,
    and bed elevation"""
    return nsidc_data.fetch(
        "BedMachineGreenland-2021-04-20.nc", downloader=_earthdata_downloader
    )


outlines_url = "https://raw.githubusercontent.com/icepack/glacier-meshes/"
outlines_commit = "5906b7c21d844a982aa012e934fe29b31ef13d41"
outlines = pooch.create(
    path=pooch.os_cache("icepack"),
    base_url=outlines_url + outlines_commit + "/glaciers/",
    registry=None,
)

registry_outlines = pkg_resources.resource_stream("icepack", "registry-outlines.txt")
outlines.load_registry(registry_outlines)


def get_glacier_names():
    r"""Return the names of the glaciers for which we have outlines that you
    can fetch"""
    return [
        os.path.splitext(os.path.basename(filename))[0]
        for filename in outlines.registry.keys()
    ]


def fetch_outline(name):
    r"""Fetch the outline of a glacier as a GeoJSON file"""
    names = get_glacier_names()
    if name not in names:
        if name == "larsen":
            warnings.warn(
                "We've added meshes of Larsen after the calving of iceberg "
                "A-68 in 2017. Please use `larsen-2015`, `larsen-2018`, or "
                "`larsen-2019` to specify the year. Returning outline for "
                "2015.",
                FutureWarning,
            )
            name = "larsen-2015"
        else:
            raise ValueError("Glacier name '%s' not in %s" % (name, names))
    downloader = pooch.HTTPDownloader(progressbar=True)
    return outlines.fetch(name + ".geojson", downloader=downloader)


def fetch_larsen_outline():
    r"""Fetch an outline of the Larsen C Ice Shelf"""
    warnings.warn(
        "This function is deprecated, use `fetch_outline('larsen')`", FutureWarning
    )
    return fetch_outline("larsen")


def fetch_mosaic_of_antarctica():
    r"""Fetch the MODIS optical image mosaic of Antarctica"""
    return nsidc_data.fetch(
        "moa750_2009_hp1_v02.0.tif.gz",
        downloader=_earthdata_downloader,
        processor=pooch.Decompress(),
    )
