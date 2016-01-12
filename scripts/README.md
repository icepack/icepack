
This directory contains code for plotting simulation results in Python using matplotlib.
The icepack testing programs (everything in `test/`) take an optional argument `-v` for verbose mode to write out the result to a `.ucd` file; the Python module `ucd.py` contains code for reading files in this format.
The module `plotting.py` contains code for plotting such a field and saving it to a `.png` file; the input and output filenames are passed as command-line arguments without extensions, i.e.

    python plotting.py velocity /home/you/velocity

will plot the results in the file `velocity.ucd` and save it to `/home/you/velocity.png`.

The scripts in this directory are mostly for my own debugging purposes and, at this juncture, are not adequate for postprocessing.
For more information on postprocessing the output of simulations performed with deal.II, see the [deal.II documentation](http://dealii.org/developer/doxygen/deal.II/group__output.html) on graphical output.
