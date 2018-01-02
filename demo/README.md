Each demo consists of a jupyter notebook.
You will need to have [jupyter](https://jupyter.org/) installed to run them.
You'll also probably be using firedrake and icepack from a virtual environment, which jupyter won't see by default.
To make the firedrake virtual environment visible within the notebook, run the following at the command line:

    source <path/to/firedrake>/firedrake/bin/active
    pip3 install ipykernel
    python3 -m ipykernel install --user --name=firedrake

From the notebook menu, you should be able to navigate to `Kernel -> Change kernel` and select `firedrake`.
(Courtesy of [PythonAnywhere](http://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs)).

Jupyter notebooks can have lots of unnecessary content; for example, metadata for which specific ipython version was used, or raw image data after running a notebook with plots.
Most of this extra content (e.g. images from plots) can be removed from the `.ipynb` file by restarting the kernel and clearing the output from the browser view of the notebook, but the extra metadata will not.
We'd rather avoid putting this information in version control.
Most of this can be stripped by using the command-line tool [jq](https://stedolan.github.io/jq/).
The following command will remove all the extra cruft:

    jq --indent 1 \
        '
        (.cells[] | select(has("outputs")) | .outputs) = []
        | (.cells[] | select(has("execution_count")) | .execution_count) = null
        | .metadata = {"language_info": {"name":"python", "pygments_lexer": "ipython3"}}
        | .cells[].metadata = {}
        ' demo.ipynb

(Courtesty of [Tim Staley](http://timstaley.co.uk/posts/making-git-and-jupyter-notebooks-play-nice/))
This should probably be made into a git commit hook, conditional on editing any of the demos.

