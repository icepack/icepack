
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='icepack',
    description='Tools for the glacier modelling library icepack',
    author='Daniel Shapero',
    author_email='shapero.daniel@gmail.com',
    url='https://gitlab.com/danshapero/icepack-py',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    packages=[
        'icepack',
        'icepack.grid',
        'icepack.mesh'
    ]
)
