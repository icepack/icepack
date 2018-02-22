
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
    install_requires=['firedrake', 'numpy', 'scipy', 'matplotlib', 'GDAL']
)
