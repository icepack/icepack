
# icepack-py

This python library contains helper functions in Python that are generally useful in conjunction with `icepack`.
These include:
* preprocessing gridded data sets into a form icepack can use
* plotting the output of finite element simulations
* manipulating and building computational meshes
and so forth.


### Installation

To install this python library, run the command

    python setup.py install --user

The extra flag `--user` will install the library under `$HOME/.local` rather than a system directory.

Note that these scripts are needed to run the pre- and post-processing scripts in the icepack examples.
