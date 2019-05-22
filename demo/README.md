Each demo consists of a jupyter notebook.
You will need to have [jupyter](https://jupyter.org/) installed to run them.
See the [installation](https://icepack.github.io/installation.html) for how to add a jupyter kernel for firedrake.

Jupyter notebooks can have lots of unnecessary content after they've been executed; for example, metadata for which specific ipython version was used, or raw image data after running a notebook with plots.
The script `sanitize.sh` will remove all this garbage and restore the notebook to its un-executed state.
You can invoke it by passing the notebook as the sole command-line argument:

    ./sanitize.sh 00-meshes-functions.ipynb

This script requires the command-line utilities [moreutils](https://joeyh.name/code/moreutils/) and [jq](https://stedolan.github.io/jq/).
The `jq` filter to clean up a notebook is courtesty of [Tim Staley](http://timstaley.co.uk/posts/making-git-and-jupyter-notebooks-play-nice/).
This should probably be made into a git commit hook, conditional on editing any of the demos.

