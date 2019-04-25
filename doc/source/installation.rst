
Installation
============

Quick start
-----------

* Install firedrake_, building PETSc with SuiteSparse along the way::

   export PETSC_CONFIGURE_OPTIONS="--download-suitesparse"
   curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
   python3 firedrake-install --install icepack

* Active the Firedrake virtual environment::

   source <path/to/firedrake>/bin/activate

* Run the icepack unit tests to make sure everything's working::

   cd $VIRTUAL_ENV/src/icepack
   pytest -s


Comments
--------

Most python projects use the simple ``python setup.py install`` formula to build and install everything.
Firedrake is appreciably more complex -- it consists of several dependent sub-packages along with a complete PETSc installation -- and thus has its own custom build process.
Rather than install the project in your system python package directory, firedrake's install script builds it inside an isolated *virtual environment*.
Python virtual environments were created to solve problems of conflicting package versions possibly breaking installed libraries.
Rather than install every python package globally, you can create an isolated virtual environment for just one package that includes a python executable and all of the package's dependencies.
This added layer of isolation keeps one package from breaking other packages on the system or doing anything that's both undesirable and hard to roll back.
However, it does introduce an annoying layer of bureaucracy in that you have to manually activate the virtual environment every time you want to use it by invoking::

   source <path/to/virtual/environment>/bin/activate

Activating a virtual environment affects only the current shell session and doesn't do anything permanent.

Firedrake uses the library PETSc_ for many of its internal data structures (meshes, vectors, matrices).
PETSc has loads of optional features, chiefly interfaces to other computational libraries.
Some of these features are mandatory for firedrake.
Rather than require you to have a PETSc installation properly configured in the way that firedrake expects, the firedrake install script builds its own version of PETSc.
This can create problems if you already do have PETSc installed on your system.
In that case, you will need to unset ``$PETSC_DIR`` and ``$PETSC_ARCH`` while installing firedrake and every time you activate the firedrake virtual environment.
While installing firedrake will fail with an error if you have a pre-existing PETSc installation, trying to run a script that uses firedrake will instead crash with a segmentation fault if you have not first unset the PETSc environment variables.

You can save yourself the trouble of remembering things by adding a function like this to your ``.bashrc`` file:

.. code-block:: bash

   firedrake-env() {
       unset PETSC_DIR PETSC_ARCH
       source <path/to/virtual/environment>/bin/activate
   }

When you type `firedrake-env` at the terminal, the PETSc environment variables from any pre-existing installation will be unset and the firedrake virtual environment will be activated.

The configuration options that firedrake uses to build PETSc do not include building the sparse direct solver UMFPACK_.
In the recommended installation instructions above, I've added an environment variable that will tell firedrake to download and link PETSc with UMFPACK in addition to the other extras.
Having a good and fast direct solver like UMFPACK is very useful for isolating errors when they happen (and they will happen).

.. _firedrake: https://www.firedrakeproject.org
.. _PETSc: https://www.mcs.anl.gov/petsc/
.. _UMFPACK: http://faculty.cse.tamu.edu/davis/suitesparse.html
