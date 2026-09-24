# Copyright (C) 2019-2025 by Daniel Shapero <shapero@uw.edu>
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

import pathlib
import requests
import pooch
import earthaccess


pooch.get_logger().setLevel("WARNING")


class EarthdataDownloader:
    def __call__(self, url, output_file, dataset):
        earthaccess.login()
        eaauth = earthaccess.__auth__
        auth = (eaauth.username, eaauth.password)
        downloader = pooch.HTTPDownloader(auth=auth, progressbar=True)
        return downloader(requests.get(url).url, output_file, dataset)


def _fetch_nsidc(search={}, **kwargs):
    kw = {
        "known_hash": None,
        "path": pooch.os_cache("icepack"),
        "downloader": EarthdataDownloader(),
        "processor": kwargs.pop("processor", None),
    }

    try:
        extra = kwargs["extra"]
        filename = kwargs["filename"]
        results = earthaccess.search_datasets(**search)
        urls = [results[0].summary["get-data"][0] + extra + filename]
    except KeyError:
        results = earthaccess.search_data(**search)
        urls = results[0].data_links()

    filenames = [
        pooch.retrieve(url, fname=pathlib.Path(url).name, **kw) for url in urls
    ]
    return filenames[0] if len(filenames) == 1 else filenames


def fetch_measures_antarctica():
    r"""Fetch the MEaSUREs Antarctic velocity map"""
    return _fetch_nsidc({"short_name": "NSIDC-0754"})


def fetch_measures_greenland():
    r"""Fetch the MEaSUREs Greenland velocity map"""
    search = {"short_name": "NSIDC-0478", "granule_name": "*200_2015_2016*"}
    return _fetch_nsidc(search)


def fetch_bedmachine_antarctica():
    r"""Fetch the BedMachine map of Antarctic ice thickness, surface elevation,
    and bed elevation"""
    return _fetch_nsidc({"short_name": "NSIDC-0756"})


def fetch_bedmachine_greenland():
    r"""Fetch the BedMachine map of Greenland ice thickness, surface elevation,
    and bed elevation"""
    search = {"short_name": "IDBMG4", "version": "6", "granule_name": "*v*.nc"}
    return _fetch_nsidc(search)


def fetch_mosaic_of_antarctica():
    r"""Fetch the MODIS optical image mosaic of Antarctica"""
    search = {"short_name": "NSIDC-0593", "version": "2"}
    filename = "moa750_2009_hp1_v02.0.tif"
    extra = "geotiff/"
    processor = pooch.Decompress(name=filename)
    kw = {"filename": filename + ".gz", "extra": extra, "processor": processor}
    return _fetch_nsidc(search, **kw)


def fetch_mosaic_of_greenland():
    r"""Fetch the MODIS optical image mosaic of Greenland"""
    search = {"short_name": "NSIDC-0547", "granule_name": "mog100_2015_hp1*.tif"}
    return _fetch_nsidc(search)


def _recursive_unzip(top_level_filename, action, pup):
    unzipper = pooch.Unzip()
    filenames = unzipper(top_level_filename, action, pup)
    results = []
    for filename in filenames:
        if pathlib.Path(filename).suffix == ".zip":
            unzipper = pooch.Unzip()
            inner_filenames = unzipper(filename, action, pup)
            results.extend(inner_filenames)

    return [r for r in results if pathlib.Path(r).suffix == ".shp"]


def fetch_randolph_glacier_inventory(name=None):
    search = {"short_name": "NSIDC-0770", "version": "7"}
    filename = "RGI2000-v7.0-G-global.zip"
    extra = "global_files/"
    kw = {"filename": filename, "extra": extra, "processor": _recursive_unzip}
    all_filenames = _fetch_nsidc(search, **kw)
    if name is None:
        return all_filenames

    filenames = [f for f in all_filenames if name in f]
    if len(filenames) == 0:
        raise ValueError("`%s` not a valid RGI region!" % name)

    return filenames[0] if len(filenames) == 1 else filenames


def get_glacier_names():
    r"""Return the names of the glaciers we have outlines for"""
    return [
        "amery", "filchner-ronne", "getz", "helheim", "hiawatha", "jakobshavn",
        "larsen-2015", "larsen-2018", "larsen-2019", "pine-island", "ross",
    ]


def fetch_outline(name, commit=None):
    r"""Fetch the outline of a glacier as a GeoJSON file"""
    if name not in get_glacier_names():
        raise ValueError("Glacier name '%s' not in %s" % (name, names))

    default_commit = "5906b7c21d844a982aa012e934fe29b31ef13d41"
    outlines_url = "https://raw.githubusercontent.com/icepack/glacier-meshes"
    url = f"{outlines_url}/{commit or default_commit}/glaciers/{name}.geojson"
    kw = {"known_hash": None, "path": pooch.os_cache("icepack"), "progressbar": True}
    return pooch.retrieve(url, fname=f"{name}.geojson", **kw)
