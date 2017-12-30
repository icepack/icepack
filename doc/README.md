
# Documentation

To build the documentation locally, run

    make html

from this directory.
This will create a website under the directory `build/html/` that you can view by navigating to
`<path/to/icepack>/doc/build/html/index.html` in your browser.

The documentation for icepack is built using [sphinx](http://www.sphinx-doc.org/en/stable/).
To build the documentation locally, you'll need sphinx, an extension for adding bibtex citations, and the Read the Docs theme.
You can install all of them by executing the following:

    pip3 install sphinx sphinxcontrib-bibtex sphinx_rtd_theme

The API documentation is built automatically using the `sphinx-apidoc` tool.
To search through the source code, this tool imports every module in icepack, which means also importing large parts of firedrake.
If you've installed firedrake in a virtual environment, you'll need to have activated the firedrake virtual environment and installed sphinx this virtual environment too.
Without first activating the firedrake virtual environment, the documentation tool will be unable to import any of the firedrake modules and in turn any of the icepack modules.

