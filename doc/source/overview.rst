
Introduction
============

Icepack is a Python library for modeling the flow of glaciers and ice sheets.
The main functionality it provides is a set of routines for solving certain nonlinear partial differential equations that describe the ice thickness, velocity, temperature, and other fields.
To implement these solvers, icepack uses the finite element analysis package `firedrake <https://firedrakeproject.org>`_.


Getting started
---------------

To use icepack, you'll need to have some familiarity with Python.
If you've never used Python, `this wikibook <https://en.wikibooks.org/wiki/Python_Programming>`_ is a good resource.
If you're mainly familiar with Matlab, `this guide <https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html>`_ will help you get acquainted with NumPy.

We do not expect that most users are already familiar with firedrake, which icepack is built on.
See the `firedrake documentation <https://firedrakeproject.org/documentation.html>`_ and especially the tutorial notebooks for an introduction.
Firedrake makes it very easy to do computational physics, even if you're not an expert in numerical methods.

See the :doc:`installation <installation>` page for how to build icepack on your computer.
From there you can read and run our tutorials to see all the things icepack can do.
The tutorials are all interactive `jupyter notebooks <https://jupyter.org/>`_.
The code samples on this website are taken directly from the demos in the icepack source directory, so you can run them interactively on your own machine too.
Some of the tutorials use real data, for which you'll also need `this repository <https://github.com/icepack/icepack-data/>`_.

For more background on glacier physics or finite element methods, we have some suggested :doc:`reading material <reading>`.


Getting involved
----------------

Icepack is free software and developed publicly on `GitHub <https://github.com/icepack/icepack>`_.
If you have a bug to report or a new feature to request, please don't hesitate to `raise an issue <https://github.com/icepack/icepack/issues>`_ on our project page.
You can also email me at the address listed on my `website <http://psc.apl.uw.edu/people/post-docs/daniel-shapero/>`_.
If you have questions that aren't answered here, or you want to get involved and don't know how, please reach out.
See the Development section for more background on the internals of icepack.
