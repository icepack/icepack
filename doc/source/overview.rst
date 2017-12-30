
Overview
========

Icepack uses the library firedrake_ for finite element analysis.
The important objects from firedrake that you'll be using are:

:py:func:`~firedrake.mesh.Mesh`
   Create the geometry for the domain

:py:func:`~firedrake.functionspace.FunctionSpace`
   Create a finite element basis over a given mesh for representing fields

:py:class:`~firedrake.function.Function`
   Represents a scalar, vector, or tensor field defined on some function space

Icepack consists of routines that solve for various quantities encountered in ice sheet physics.
For example, the class :py:class:`~icepack.models.ice_shelf.IceShelf` has methods :py:meth:`~icepack.models.ice_shelf.IceShelf.diagnostic_solve` and :py:meth:`~icepack.models.ice_shelf.IceShelf.prognostic_solve` that compute (respectively) the ice velocity and thickness.
The computed velocity and thickness are `firedrake.Function` objects that you can plot or analyze using the various routines in firedrake.

For a thorough reference on ice sheet physics, see :cite:`greve2009dynamics` and :cite:`cuffey2010physics`.
For an introduction to the finite element method, see :cite:`gockenbach2006understanding` and :cite:`braess2007finite`.


.. _firedrake: https://www.firedrakeproject.org
.. bibliography:: references.bib
   :style: unsrt

