# icepack
Finite element modeling of glaciers and ice sheets

This library is for simulating the flow of glaciers and ice sheets using the finite element method.

Icepack is currently in active development and is not yet ready for use.

Planned features:
* multiple glacier models: shallow shelf, full Stokes, depth-averaged higher-order models
* prognostic modeling of ice thickness and temperature evolution
* inverse methods for basal shear stress, ice viscosity


Compilation & installation
==========================

You will first need a working deal.II installation and the environment variable `DEAL_II_DIR` must be set to the directory of your deal.II installation.

To build the icepack sources, run the following:
```
mkdir <build>
cd <build>
cmake <path/to/icepack>
make
```

Additionally, unit tests can be run by invoking `make test`.


Dependencies
============

You will need the following packages installed in order to use icepack:

* a C++11 compiler, e.g. clang 3.2+, GCC 4.7+, icc 12+
* [CMake](http://www.cmake.org/) 2.8.11+. Both deal.II and icepacke use the CMake build system.
* [deal.II](http://dealii.org/) 8.2.1+. General-purpose finite element library on which icepack is built.
* [GDAL](http://www.gdal.org/). Geospatial Data Abstraction Library, for reading geotif and other similar file formats.
* lapack, blas

Although not strictly necessary, you will also probably want to have:
* Python, with numpy, scipy, matplotlib
* [OpenMPI](http://www.open-mpi.org/) The Message-Passing Interface, for distributed-memory parallel computations. While applications that use MPI for parallelism should not depend on the specific implementation (OpenMPI, MPICH, ...), this is unfortunately not always the case.
* [PETSc](http://www.mcs.anl.gov/petsc/) 3.5+. The Portable Extensible Toolkit for Scientific Computation, used for sparse matrix solvers.
* [parmetis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) Graph partitioning library, for parallel computations.
* [valgrind](http://valgrind.org/) memory debugger

Additionally, while the GCC compiler usually generates faster assembly code, we recommend using Clang for development purposes.
Icepack uses the deal.II library for finite element computations, which relies quite heavily on C++ templates.
Clang gives far more helpful error messages for mistakes in code than GCC does, especially when debugging improper use of templates.
Additionally, the clang memory and address sanitizers can find subtle bugs easily.
Configuring the MPI wrapper compilers to use clang instead of GCC is [straightforward](http://stackoverflow.com/questions/14464554/is-there-an-easy-way-to-use-clang-with-open-mpi).
