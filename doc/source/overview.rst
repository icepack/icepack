
Overview
========

Icepack is a Python library for modeling the flow of glaciers and ice sheets.
The main functionality it provides is a set of routines for solving certain nonlinear partial differential equations that describe the ice thickness, velocity, temperature, and other fields.
To implement these solvers, icepack uses the finite element analysis package `firedrake <https://firedrakeproject.org>`_.

To use icepack, you'll need to have some familiarity with Python.
If you've never used Python, `this wikibook <https://en.wikibooks.org/wiki/Python_Programming>`_ is a good resource.
If you're mainly familiar with Matlab, `this guide <https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html>`_ will help you get acquainted with NumPy.

We do not expect that most users are already familiar with firedrake, which icepack is built on.
See the `firedrake documentation <https://firedrakeproject.org/documentation.html>`_ and especially the tutorial notebooks for an introduction.
Firedrake makes it very easy to do computational physics, even if you're not an expert in numerical methods.

See the :doc:`installation <installation>` page for how to build icepack on your computer.
From there you can read and run our tutorials to see all the things icepack can do.
The tutorials are interactive `jupyter notebooks <https://jupyter.org/>`_.
The code samples on this website are taken directly from the demos in the icepack source directory, so you can run them interactively on your own machine too.
Some of the tutorials use real data, for which you'll need an account with `NASA EarthData <https://urs.earthdata.nasa.gov/>`_.
