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

r"""Routines for fetching the glaciological data sets used in the demos"""

import os
import pkg_resources
from getpass import getpass
import requests
import pooch

pooch.get_logger().setLevel('WARNING')

def _earthdata_downloader(url, output_file, dataset):
    username = os.environ.get('EARTHDATA_USERNAME')
    if username is None:
        username = input('EarthData username: ')

    password = os.environ.get('EARTHDATA_PASSWORD')
    if password is None:
        password = getpass('EarthData password: ')
    auth = (username, password)

    login = requests.get(url)
    downloader = pooch.HTTPDownloader(auth=auth, progressbar=True)
    try:
        downloader(login.url, output_file, dataset)
    except requests.exceptions.HTTPError as error:
        if 'Unauthorized' in str(error):
            pooch.get_logger().error('Wrong username/password!')
        raise error


nsidc_data = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url='',
    registry=None
)

registry_file = pkg_resources.resource_stream('icepack', 'registry.txt')
nsidc_data.load_registry(registry_file)


def fetch_measures_antarctica():
    return nsidc_data.fetch(
        'antarctic_ice_vel_phase_map_v01.nc', downloader=_earthdata_downloader
    )


def fetch_measures_greenland():
    return [
        nsidc_data.fetch(
            'greenland_vel_mosaic200_2015_2016_{}_v02.1.tif'.format(field_name),
            downloader=_earthdata_downloader
        )
        for field_name in ['vx', 'vy', 'ex', 'ey']
    ]


def fetch_bedmachine_antarctica():
    return nsidc_data.fetch(
        'BedMachineAntarctica_2019-11-05_v01.nc',
        downloader=_earthdata_downloader
    )


outlines_url = 'https://raw.githubusercontent.com/icepack/glacier-meshes/'
outlines_commit = 'a522188dadb9ba49d4848ba66cab8c90f9fda5d9'
larsen_outline = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url=outlines_url + outlines_commit + '/glaciers/',
    registry={
        'larsen.geojson':
        'da77c1920191d415961347b43e18d5bc2ffd72ddb803c01fc24c68c5db0f3033'
    }
)


def fetch_larsen_outline():
    downloader = pooch.HTTPDownloader(progressbar=True)
    return larsen_outline.fetch('larsen.geojson', downloader=downloader)


def fetch_mosaic_of_antarctica():
    return nsidc_data.fetch(
        'moa750_2009_hp1_v01.1.tif.gz',
        downloader=_earthdata_downloader,
        processor=pooch.Decompress()
    )
