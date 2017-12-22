
from glob import glob
from os.path import basename, splitext
from setuptools import setup, find_packages

setup(
    name='icepack',
    version='0.1.0',
    license='GPL v3',
    description='ice sheet flow modelling with the finite element method',
    author='Daniel Shapero',
    url='https://github.com/icepack/icepack',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')]
)

