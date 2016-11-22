# icepack
Finite element modeling of glaciers and ice sheets

This library is for simulating the flow of glaciers and ice sheets using the finite element method.
The core library code is contained in this repository; examples and tutorials are available in a separate repo [here](https://github.com/danshapero/icepack-examples).

Icepack is not yet ready for use and should be considered pre-alpha quality.
The code is not optimized at all.
The nonlinear solvers are not robust.

Planned features:

* multiple glacier models: full Stokes, depth-averaged higher-order models
* prognostic modeling of temperature evolution
* inverse methods for basal shear stress


## Compilation & installation

You will first need a working deal.II installation.
If deal.II is not installed in some standard directory, e.g. `/usr` or `/usr/local`, the environment variable `DEAL_II_DIR` can be set to the directory of your deal.II installation.

We also rely on several C++11 features -- lambda, move semantics, etc. -- so you will need to compile deal.II with C++11 support.
When configuring deal.II with CMake, add the flag `-DDEAL_II_WITH_CXX11:BOOL=True`.
If your compiler supports C++11, this should be detected automatically, but adding this flag will make sure.

For the time being, icepack relies on the sparse direct linear algebra solver [UMFPACK](http://faculty.cse.tamu.edu/davis/suitesparse.html), so deal.II must be configured to use UMFPACK.
When configuring deal.II with CMake, the flag `-DDEAL_II_WITH_UMFPACK:BOOL=True` must be added to enable UMFPACK.

To build the icepack sources, run the following:

    mkdir <build>
    cd <build>
    cmake <path/to/icepack>
    make

Unit tests can be run by invoking `make test`.

The [examples repository](https://github.com/danshapero/icepack-examples) contains example programs demonstrating the use of icepack for real applications.
These include pre- and post-processing scripts in python.
A helper library is included in the directory `python/`, which will be built and installed automatically along with the rest of icepack.


## Dependencies

You will need the following packages installed in order to use icepack:

* a C++11 compiler, e.g. clang 3.2+, GCC 4.7+, icc 12+
* [CMake](http://www.cmake.org/) 2.8.11+. Both deal.II and icepacke use the CMake build system.
* [deal.II](http://dealii.org/) development branch. General-purpose finite element library on which icepack is built.
* [GDAL](http://www.gdal.org/). Geospatial Data Abstraction Library, for reading geotif and other similar file formats.
* lapack, blas

To build and run the example programs, you will also need:

* Python 3
* numpy, scipy
* matplotlib

Although not strictly necessary, you will also probably want to have:

* [OpenMPI](http://www.open-mpi.org/) The Message-Passing Interface, for distributed-memory parallel computations. While applications that use MPI for parallelism should not depend on the specific implementation (OpenMPI, MPICH, ...), this is unfortunately not always the case.
* [PETSc](http://www.mcs.anl.gov/petsc/) 3.5+. The Portable Extensible Toolkit for Scientific Computation, used for sparse matrix solvers.
* [parmetis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) Graph partitioning library, for parallel computations.
* [valgrind](http://valgrind.org/) memory debugger

For development purposes, you may want to disable the use of Intel's Threading Building Blocks library when building deal.II by passing the flag `-DDEAL_II_WITH_THREADS:BOOL=False` to `cmake`.
Using Intel TBB can cause valgrind to erroneously report memory leaks, and to generally confound debuggers.

