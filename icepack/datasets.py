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


nsidc_url = 'https://daacdata.apps.nsidc.org/pub/DATASETS/'

measures_antarctica = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url=nsidc_url + 'nsidc0754_MEASURES_antarctic_ice_vel_phase_map_v01/',
    registry={
        'antarctic_ice_vel_phase_map_v01.nc':
        'fa0957618b8bd98099f4a419d7dc0e3a2c562d89e9791b4d0ed55e6017f52416'
    }
)

def fetch_measures_antarctica():
    return measures_antarctica.fetch('antarctic_ice_vel_phase_map_v01.nc',
                                     downloader=_earthdata_downloader)


measures_greenland = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url=nsidc_url + 'NSIDC-0478.002/2015.0.01/',
    registry={
        'greenland_vel_mosaic200_2015_2016_vx_v02.1.tif':
        '77b8eb65a4718da055bb048b75c35b48ca43d76b5bffb650932128d60ed28598',
        'greenland_vel_mosaic200_2015_2016_vy_v02.1.tif':
        'fb5dbc07d032de9b1bdb0b990ed02a384964d73a150529515038139efb1e3193',
        'greenland_vel_mosaic200_2015_2016_ex_v02.1.tif':
        '7e980fb7845fb8517f3791c9b3912ac62d3ce760938d062f1a4d575fad02ad89',
        'greenland_vel_mosaic200_2015_2016_ey_v02.1.tif':
        '9a43092b4c92ac767dbc4e6b7d42e55887420bec94eb654382d724d2a2ab6d9a'
    }
)

def fetch_measures_greenland():
    return [
        measures_greenland.fetch(
            'greenland_vel_mosaic200_2015_2016_{}_v02.1.tif'.format(field_name),
            downloader=_earthdata_downloader
        )
        for field_name in ['vx', 'vy', 'ex', 'ey']
    ]


bedmap2 = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url='https://secure.antarctica.ac.uk/data/bedmap2/',
    registry={
        'bedmap2_tiff.zip':
        'f4bb27ce05197e9d29e4249d64a947b93aab264c3b4e6cbf49d6b339fb6c67fe'
    }
)

def fetch_bedmap2():
    downloader = pooch.HTTPDownloader(progressbar=True)
    filenames = bedmap2.fetch(
        'bedmap2_tiff.zip', processor=pooch.Unzip(), downloader=downloader
    )
    return [f for f in filenames if os.path.splitext(f)[1] == '.tif']


bedmachine_antarctica = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url='https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0756.001/1970.01.01/',
    registry={
        'BedMachineAntarctica_2019-11-05_v01.nc':
        '06a01511a51bbc27d5080e4727a6523126659fe62402b03654a5335e25b614c0'
    }
)

def fetch_bedmachine_antarctica():
    return bedmachine_antarctica.fetch('BedMachineAntarctica_2019-11-05_v01.nc',
                                       downloader=_earthdata_downloader)


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
    return larsen_outline.fetch(
        'larsen.geojson', downloader=pooch.HTTPDownloader(progressbar=True)
    )


moa = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url=nsidc_url + 'nsidc0593_moa2009/geotiff/',
    registry={
        'moa750_2009_hp1_v01.1.tif.gz':
        '90d1718ea0971795ec102482c47f308ba08ba2b88383facb9fe210877e80282c'
    }
)

def fetch_mosaic_of_antarctica():
    return moa.fetch(
        'moa750_2009_hp1_v01.1.tif.gz',
        downloader=_earthdata_downloader,
        processor=pooch.Decompress()
    )
