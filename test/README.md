
This directory contains unit tests for icepack.
To test the code, run the command

    PYTHONPATH=$PWD py.test

from the icepack source directory.
If you have the [`pytest-cov`](https://pypi.python.org/pypi/pytest-cov) extension, you can generate a test coverage report by running

    PYTHONPATH=$PWD py.test --cov-report html --cov icepack --verbose

which you can view by opening the file `htmlcov/index.html` in a web browser.

