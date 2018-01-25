
# Documentation

You'll probably be using firedrake and icepack from a virtual environment; you'll need to have the virtual environment activated and have sphinx installed in it in order to build the documentation.
Once you've activated the virtual environment, all the dependencies for building the documentation can be installed by running

    pip3 install sphinx sphinxcontrib-bibtex sphinx_rtd_theme

at the command line.

To build the documentation locally, run

    make html

from this directory.
This will create a website under the directory `build/html/` that you can view by navigating to
`<path/to/icepack>/doc/build/html/index.html` in your browser.

