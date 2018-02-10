
import sys
import os.path

text = r"""
icepack
=======

Welcome to the documentation for `icepack`, a python library for modeling the flow of ice sheets and glaciers!
The main design goals for icepack are:

interactivity
  You can run simulations and visualize the results from the python interpreter.

robustness
  The numerical solvers should "just work" with little or no tweaking on the part of you, the user.

transparency
  Each physics solver returns the relevant field and you can analyze or post-process it any way you want.

extensibility
  Different parts of the model physics are relatively orthogonal to each other, and can be swapped out with little effort.



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   overview.rst
   installation.rst

.. toctree::
   :maxdepth: 1
   :caption: Demos:

{0}

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   modules.rst

Indices and tables
==================

* :ref:`genindex`
"""

if __name__ == "__main__":
    filenames = [os.path.basename(arg).replace("ipynb", "rst")
                 for arg in sys.argv[1:]]
    file_list = "".join(["   icepack.demo.{}\n".format(f) for f in filenames])

    with open("source/index.rst", 'w') as index:
        index.write(text.format(file_list))
