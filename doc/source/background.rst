Background
==========

Understanding the tutorials that follow will require some knowledge of glacier physics.
In particular, we'll expect that you have a basic understanding of:

* mass, momentum, and heat transport
* the Stokes equations and simplified models
* Glen's flow law

We use a slightly unconventional unit system borrowed from the package `Elmer/Ice <http://elmerice.elmerfem.org/>`_.
The fundamental units are megapascals (MPa), meters (m) and years (yr).
In this unit system, most of the quantities we're interested in, including physical constants, are between 1/10 and 1000.

Symbols
-------

In the tutorials and in the icepack API we've tried to use variable names that correspond as closely to the symbols used for each field in the literature and standard textbooks.
Our goal is to make the code look as much like the mathematics as possible.
The variable names and corresponding fields are:

* **u** - velocity
* **h** - thickness
* **s** - surface elevation
* **b** - bed elevation
* **A** - rheology coefficient
* **n** - exponent in Glen's flow law
* **C** - friction coefficient
* **m** - exponent in Weertman's sliding law
* **E** - internal energy density

Many references in the glaciological literature use a lower-case :math:`h` for surface elevation and a capital :math:`H` for thickness.
We've chosen a slightly different convention, so this is something to be aware of.


Reading
-------

The standard textbook on glaciology is *The Physics of Glaciers* by Cuffey and Paterson :cite:`cuffey2010physics`.
This book is great for breadth and covers many more topics than modeling.
*Dynamics of Ice Sheets and Glaciers* by Greve and Blatter :cite:`greve2009dynamics` focuses more on continuum mechanics and modeling and has some nice introductions to various numerical methods.
While a thorough knowledge of the finite element method shouldn't be necessary to use icepack, some familiarity is definitely helpful.
*Understanding and Implementing the Finite Element Method* by Gockenbach is a good introduction :cite:`gockenbach2006understanding`.
*Finite Elements: Theory, Fast Solvers, and Applications in Solid Mechanics* by Braess goes into much greater depth :cite:`braess2007finite`.


.. bibliography:: references.bib
   :style: unsrt
