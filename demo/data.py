import os
from getpass import getpass
import subprocess
import requests
import pooch

measures_antarctica = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url='https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0484.002/1996.01.01/',
    registry={
        'antarctica_ice_velocity_450m_v2.nc':
        '268be94e3827b9b8137b4b81e3642310ca98a1b9eac48e47f91d53c1b51e4299'
    }
)

def _earthdata_downloader(url, output_file, dataset):
    username = os.environ.get('EARTHDATA_USERNAME')
    if username is None:
        username = input('EarthData username: ')

    password = os.environ.get('EARTHDATA_PASSWORD')
    if password is None:
        password = getpass('EarthData password: ')

    login = requests.get(url)
    downloader = pooch.HTTPDownloader(auth=(username, password))
    downloader(login.url, output_file, dataset)

def fetch_measures_antarctica():
    return measures_antarctica.fetch('antarctica_ice_velocity_450m_v2.nc',
                                     downloader=_earthdata_downloader)

bedmap2 = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url='https://secure.antarctica.ac.uk/data/bedmap2/',
    registry={
        'bedmap2_tiff.zip':
        'f4bb27ce05197e9d29e4249d64a947b93aab264c3b4e6cbf49d6b339fb6c67fe'
    }
)

def fetch_bedmap2():
    filenames = bedmap2.fetch('bedmap2_tiff.zip', processor=pooch.Unzip())
    return [f for f in filenames if os.path.splitext(f)[1] == '.tif']


larsen_outline = pooch.create(
    path=pooch.os_cache('icepack'),
    base_url='https://raw.githubusercontent.com/icepack/glacier-meshes/master/glaciers/',
    registry={
        'larsen.geojson':
        '06a0ae21a3a55a8391ff32eefb69e0e0ec8f29bca68d9bc4cd23abb5010397f8'
    }
)

def fetch_larsen_outline():
    return larsen_outline.fetch('larsen.geojson')
