Each demo consists of a jupyter notebook.
You will need to have [jupyter](https://jupyter.org/) installed to run them.
You'll also probably be using firedrake and icepack from a virtual environment, which jupyter won't see by default.
To make the firedrake virtual environment visible within the notebook, run the following at the command line:

    source <path/to/firedrake>/firedrake/bin/active
    pip3 install ipykernel
    python3 -m ipykernel install --user --name=firedrake

From the notebook menu, you should be able to navigate to `Kernel -> Change kernel` and select `firedrake`.
(Courtesy of [PythonAnywhere](http://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs)).

Jupyter notebooks can have lots of unnecessary content after they've been executed; for example, metadata for which specific ipython version was used, or raw image data after running a notebook with plots.
The script `sanitize.sh` will remove all this garbage and restore the notebook to its un-executed state.
You can invoke it by passing the notebook as the sole command-line argument:

    ./sanitize.sh 00-meshes-functions/meshes.ipynb

This script requires the command-line utilities [moreutils](https://joeyh.name/code/moreutils/) and [jq](https://stedolan.github.io/jq/).
The `jq` filter to clean up a notebook is courtesty of [Tim Staley](http://timstaley.co.uk/posts/making-git-and-jupyter-notebooks-play-nice/).
This should probably be made into a git commit hook, conditional on editing any of the demos.

