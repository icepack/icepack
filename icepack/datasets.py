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


pooch.get_logger().setLevel('WARNING')


class EarthDataDownloader:
    def __init__(self):
        self._username = None
        self._password = None

    def _get_credentials(self):
        if self._username is None:
            username_env = os.environ.get('EARTHDATA_USERNAME')
            if username_env is None:
                self._username = input('EarthData username: ')
            else:
                self._username = username_env

        if self._password is None:
            password_env = os.environ.get('EARTHDATA_PASSWORD')
            if password_env is None:
                self._password = getpass('EarthData password: ')
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
            if 'Unauthorized' in str(error):
                pooch.get_logger().error('Wrong username/password!')
                self._username = None
                self._password = None
            raise error


_earthdata_downloader = EarthDataDownloader()


nsidc_data = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url='',
    registry=None
)

registry_nsidc = pkg_resources.resource_stream('icepack', 'registry-nsidc.txt')
nsidc_data.load_registry(registry_nsidc)


def fetch_measures_antarctica():
    r"""Fetch the MEaSUREs Antarctic velocity map"""
    return nsidc_data.fetch(
        'antarctic_ice_vel_phase_map_v01.nc', downloader=_earthdata_downloader
    )


def fetch_measures_greenland():
    r"""Fetch the MEaSUREs Greenland velocity map"""
    return [
        nsidc_data.fetch(
            'greenland_vel_mosaic200_2015_2016_{}_v02.1.tif'.format(field_name),
            downloader=_earthdata_downloader
        )
        for field_name in ['vx', 'vy', 'ex', 'ey']
    ]


def fetch_bedmachine_antarctica():
    r"""Fetch the BedMachine map of Antarctic ice thickness, surface elevation,
    and bed elevation"""
    return nsidc_data.fetch(
        'BedMachineAntarctica_2020-07-15_v02.nc',
        downloader=_earthdata_downloader
    )


outlines_url = 'https://raw.githubusercontent.com/icepack/glacier-meshes/'
outlines_commit = 'c98a8b7536b1891611566257d944e5ea024f2cdf'
outlines = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url=outlines_url + outlines_commit + '/glaciers/',
    registry=None
)

registry_outlines = pkg_resources.resource_stream(
    'icepack', 'registry-outlines.txt'
)
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
        raise ValueError("Glacier name '%s' not in %s" % (name, names))
    downloader = pooch.HTTPDownloader(progressbar=True)
    return outlines.fetch(name + '.geojson', downloader=downloader)


def fetch_larsen_outline():
    r"""Fetch an outline of the Larsen C Ice Shelf"""
    warnings.warn(
        "This function is deprecated, use `fetch_outline('larsen')`",
        FutureWarning
    )
    return fetch_outline('larsen')


def fetch_mosaic_of_antarctica():
    r"""Fetch the MODIS optical image mosaic of Antarctica"""
    return nsidc_data.fetch(
        'moa750_2009_hp1_v01.1.tif.gz',
        downloader=_earthdata_downloader,
        processor=pooch.Decompress()
    )
