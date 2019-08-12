
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

* Run one of the icepack unit tests to make sure it works::

   cd $VIRTUAL_ENV/src/icepack
   pytest -s test/ice_shelf_test.py

Run the demos
-------------

* Install the mesh generator gmsh_.
  Binaries are available on their website, or you can use your system package manager.
  On MacOS::

   brew install gmsh

  On Debian, Mint, or Ubuntu::

   sudo apt-get install gmsh

* Make a jupyter kernel for firedrake::

   pip3 install ipykernel
   python3 -m ipykernel install --user --name=firedrake

* Run the demo notebooks::

   cd $VIRTUAL_ENV/src/icepack/demo
   jupyter notebook


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

You can save yourself the trouble of remembering things by adding a function like this to your ``.bashrc`` or ``.bash_profile``:

.. code-block:: bash

   firedrake-env() {
       source <path/to/virtual/environment>/bin/activate
   }

When you type `firedrake-env` at the terminal, the firedrake virtual environment will be activated.
If you find yourself using lots of different virtual environments, you might also like virtualenvwrapper_.

Firedrake uses the library PETSc_ for many of its internal data structures (meshes, vectors, matrices).
PETSc has loads of optional features, chiefly interfaces to other computational libraries.
Some of these features are mandatory for firedrake.
Rather than require you to have a PETSc installation properly configured in the way that firedrake expects, the firedrake install script builds its own version of PETSc.
This can create problems if you already do have PETSc installed on your system.
In that case, you will need to unset ``$PETSC_DIR`` and ``$PETSC_ARCH`` while installing firedrake and every time you activate the firedrake virtual environment.
You can add an extra line to the `firedrake-env` command above to unset these variables if you want to keep a pre-existing PETSc installation and also use firedrake.
While installing firedrake will fail with an error if you have a pre-existing PETSc installation, trying to run a script that uses firedrake will instead crash with a segmentation fault if you have not first unset the PETSc environment variables.

The configuration options that firedrake uses to build PETSc do not include building the sparse direct solver UMFPACK_.
In the recommended installation instructions above, I've added an environment variable that will tell firedrake to download and link PETSc with UMFPACK in addition to the other extras.
Having a good and fast direct solver like UMFPACK is very useful for isolating errors when they happen (and they will happen).

To run all of the demos, you'll need to have the mesh generator gmsh_.
Gmsh will take a description of the outline of some domain in 2D and calculate a triangulation of the interior.
Since the shape of most real glaciers is very irregular and thus can't easily be represented by a rectilinear grid, a tool like gmsh is indispensible.
Gmsh is available through the package managers of most operating systems or as a binary executable from the project website.

.. _gmsh: http://gmsh.info/
.. _firedrake: https://www.firedrakeproject.org
.. _PETSc: https://www.mcs.anl.gov/petsc/
.. _UMFPACK: http://faculty.cse.tamu.edu/davis/suitesparse.html
.. _virtualenvwrapper: https://virtualenvwrapper.readthedocs.io/en/latest/
