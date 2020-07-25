
Engineering
===========

If you've taken a university class in scientific programming or numerical methods, you probably learned how to write code that would fulfill a particular task correctly and efficiently.
The level of granularity in these sorts of classes is usually a single main function and possibly some additional helper functions.
To succeed, you might have to know how to derive good finite difference formulas, a couple of timestepping schemes for ODEs, a couple of different methods to solve linear systems, how to loop efficiently over a multidimensional array in your language of choice, how to parallelize a function using OpenMP or MPI, and maybe a bit about caches.

What these courses rarely cover is how to design and maintain larger software packages.
What are the things that make a software package useful for the target audience and what about the structure or design can make it more or less useful?
What can make it more or less difficult to maintain for the developers?
These concerns fall under the aegis of *software engineering*.
Part of the motivation for developing icepack in the first place was to do new and exciting things with the physics of glacier flow modeling.
An equally large part -- and one of the main reasons why NSF and NASA have elected to fund this project -- was our commitment to good software engineering.
Here we'll highlight a few of the principles that have guided the design of icepack.
The idea of the `cognitive dimensions`_ of notations from the field of human-computer interaction has been a big inspiration in how we have organized the library.

Feedback
--------

How easily can you inspect or query the working state of the system?
Programming in Python and, especially, working from the interpreter or from Jupyter notebooks makes it easy to obtain diagnostics about the state of a simulation.
Icepack has been designed to offer a fairly fine level of granularity to you (the user) in how each task is executed.
For example, rather than run an entire flow model all at once from the beginning to the end time, we provide the functions to do diagnostic and prognostic model solves which you can then string together in a loop.
By focusing more on these relatively lower-level primitives and leaving it to you to write the loop, we leave open the possibility for you to insert whatever diagnostic code you want in that loop.
At an even lower level, you can manually step through the simulation one interval at a time.
Similarly, the inverse solver provides a method to keep iterating to convergence, but it also provides a method to perform one iteration at a time.

Closeness
---------

How well does the "notation" or interface that the software provides correspond to the idealized problem domain?
The two main components of an ice flow model are the diagnostic and prognostic physics, which govern respectively the ice velocity as a function of slope and how that slope evolves as a function of velocity and accumulation.
The core of any simulation with icepack consists of a loop that alternates calls to functions that solve these two problems.
In this way the structure of the code for a simulation mirrors the mental model that many glaciologists have of how ice flow works.

Firedrake is another good example of improving on closeness of mapping.
The code to express the weak form of a partial differential equation using Firedrake often looks nearly identical to the math.
In lower-level libraries, the code that a modeler would write consists of deeply-nested loops to fill matrices and vectors, and the connection to the underlying math can easily be lost.

Consistency
-----------

This attribute describes how easy it is, given familiarity with part of the software, to guess the functioning or behavior of the remaining part.
For example:

* the physics models in icepack take in all their arguments by keyword
* the diagnostic models take in a velocity, thickness, and rheology
* the prognostic models take in a timestep, a thickness, a velocity, and an accumulation rate

If you're already familiar with using, say, the ice shelf model, the functioning of the ice stream model is similar but for the fact that you have to also pass in the surface elevation field and calculate it from the thickness at each step.
If you're already familiar with using the shallow stream model, the hybrid model works in the same way, but you have to create different function spaces.
Some differences are inevitable because different models have a different number of fields or are posed on a 3D rather than a 2D geometry.
But the mechanics of what goes into these models and how is largely the same in each case.

Modularity
----------

Distinct parts of the software should be responsible for different things and the details of one should not affect the correct functioning of another.
For example, in both Firedrake and in icepack, expressing a given computational problem requires two parts.
First, you have to create a "problem" or "model" object which specifies *what* you are doing, and only then do you create a "solver" object which specifies *how* to actually do it.
This design is evident in the Firedrake linear variational problem and solver objects, the icepack model and flow solver classes, and in the icepack inverse problem and solver objects.
A glaciologist interested in experimenting with the physics of ice flow would be focusing on using the model objects to formulate new problems, and how the resulting nonlinear system of equations gets solved doesn't matter as long as it's correct.
By contrast, an applied mathematician interested in performance optimization of PDE solvers would be focusing on the solver objects for existing and well-studied models.

.. _cognitive dimensions: https://en.wikipedia.org/wiki/Cognitive_dimensions_of_notations
