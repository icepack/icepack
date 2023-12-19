# Copyright (C) 2019-2023 by Daniel Shapero <shapero@uw.edu>
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
import requests
import pooch


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


_n5eil01u = "https://n5eil01u.ecs.nsidc.org"
_daacdata = "https://daacdata.apps.nsidc.org/pub/DATASETS"
_nsidc_links = {
    "antarctic_ice_vel_phase_map_v01.nc": (
        "md5:2e1ca76870a6e67ace309a9850739dc9",
        f"{_n5eil01u}/MEASURES/NSIDC-0754.001/1996.01.01",
    ),
    "BedMachineAntarctica_2020-07-15_v02.nc": (
        "md5:35b36e1527fd846cbf38ce25b0e0c563",
        f"{_n5eil01u}/MEASURES/NSIDC-0756.002/1970.01.01",
    ),
    "BedMachineGreenland-v5.nc": (
        "md5:7387182a059dd8cad66ce7638eb0d7cd",
        f"{_n5eil01u}/ICEBRIDGE/IDBMG4.005/1993.01.01",
    ),
    "moa750_2009_hp1_v02.0.tif.gz": (
        "md5:7d386e916cbc072cd3ada4ee3ba145c9",
        f"{_daacdata}/nsidc0593_moa2009_v02/geotiff",
    ),
    "greenland_vel_mosaic200_2015_2016_vx_v02.1.tif": (
        "md5:48bfa5266b6ecf5d4939c306f665ce47",
        f"{_n5eil01u}/MEASURES/NSIDC-0478.002/2015.09.01",
    ),
    "greenland_vel_mosaic200_2015_2016_vy_v02.1.tif": (
        "md5:f68a5bbc76bcbb11b3cfe7a979d64651",
        f"{_n5eil01u}/MEASURES/NSIDC-0478.002/2015.09.01",
    ),
    "greenland_vel_mosaic200_2015_2016_ex_v02.1.tif": (
        "md5:e9e3d01d630533d870d552da023a66ba",
        f"{_n5eil01u}/MEASURES/NSIDC-0478.002/2015.09.01",
    ),
    "greenland_vel_mosaic200_2015_2016_ey_v02.1.tif": (
        "md5:1d1b5b0efcdf24218e9f7d75b6750a3d",
        f"{_n5eil01u}/MEASURES/NSIDC-0478.002/2015.09.01",
    ),
    "RGI2000-v7.0-G-01_alaska.zip": (
        "md5:dcde7c544799aff09ad9ea11616fa003",
        f"{_daacdata}/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-G",
    ),
}
nsidc_data = pooch.create(
    path=pooch.os_cache("icepack"),
    base_url="",
    registry={name: md5sum for name, (md5sum, url) in _nsidc_links.items()},
    urls={name: f"{url}/{name}" for name, (md5sum, url) in _nsidc_links.items()},
)


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
        "BedMachineGreenland-v5.nc", downloader=_earthdata_downloader
    )


_outlines_url = "https://raw.githubusercontent.com/icepack/glacier-meshes"
_outlines_commit = "5906b7c21d844a982aa012e934fe29b31ef13d41"
outlines = pooch.create(
    path=pooch.os_cache("icepack"),
    base_url=f"{_outlines_url}/{_outlines_commit}/glaciers/",
    registry={
        "amery.geojson": "md5:b9a32abaacc3a36d5b696a26c2bd1b9b",
        "filchner-ronne.geojson": "md5:7876e9fad2fe74a99f3b1ff92e12dc3c",
        "getz.geojson": "md5:31dc3f10c0a06c05020683e8cb5a9f59",
        "helheim.geojson": "md5:21b754c088ceeb5995295a6ce54783e0",
        "hiawatha.geojson": "md5:3b0aa71d21641792b1bbbda35e185cca",
        "jakobshavn.geojson": "md5:baf707914993fb052e00024ccdceab92",
        "larsen-2015.geojson": "md5:317ba73b8a2370ec0832b0bc0bcfc986",
        "larsen-2018.geojson": "md5:cccb22fd94143d6ccbb4aaa08dee6cad",
        "larsen-2019.geojson": "md5:3188635279f93e863ae800fecb9d085a",
        "pine-island.geojson": "md5:2ebfb7a321568dcd481771ab3f0993c6",
        "ross.geojson": "md5:a4cf6461607c90961280e5afbab1123b",
    },
)


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
        raise ValueError("Glacier name '%s' not in %s" % (name, names))
    downloader = pooch.HTTPDownloader(progressbar=True)
    return outlines.fetch(name + ".geojson", downloader=downloader)


def fetch_randolph_glacier_inventory(region):
    r"""Fetch a regional segment of the Randolph Glacier Inventory"""
    downloader = _earthdata_downloader
    filenames = nsidc_data.fetch(
        f"RGI2000-v7.0-G-01_{region}.zip",
        downloader=_earthdata_downloader,
        processor=pooch.Unzip(),
    )
    return [f for f in filenames if ".shp" in f][0]


def fetch_mosaic_of_antarctica():
    r"""Fetch the MODIS optical image mosaic of Antarctica"""
    return nsidc_data.fetch(
        "moa750_2009_hp1_v02.0.tif.gz",
        downloader=_earthdata_downloader,
        processor=pooch.Decompress(),
    )
