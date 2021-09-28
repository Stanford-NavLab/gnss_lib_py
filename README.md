# gnss_lib_py
Python code used for processing and simulating GNSS measurements.

## Setup
Note: Written and tested for Ubuntu 18 and 20 (standalone) and Ubuntu 20 on WSL2 

Note: Written with VS Code in mind

## Installation instructions for New Users
1. Follow the installation section [here for Windows](https://pypi.org/project/pyenv-win/) or [here for Linux/WSL2](https://github.com/pyenv/pyenv#installation). After completing steps 1-3 for Windows, or 1-5 for Linux,  install Python using `pyenv install 3.8.9+`.
2. Finish the installation using the remaining steps in the `pyenv` documentation. 
3. Install `poetry` using the instructions [here](https://python-poetry.org/docs/#installation).
4. Clone our repository. Using the command `pyenv local` and verify that the Python version is `3.8.9`.
5. Install dependencies using `poetry install`.
6. If using Jupyter Notebooks and they don't work: Setup Jupyter as mentioned [here](https://blog.jayway.com/2019/12/28/pyenv-poetry-saviours-in-the-python-chaos/) using the command `poetry run ipython kernel install --user --name=gnss_lib_py`.
7. To run the Poetry shell, use `poetry shell`. To run files using the local poetry environment, use `poetry run python *.py`.

## Running notebooks with VS Code
Run `poetry shell` to activate the environment. Then run `code .` in order to open VS Code within the environment. At the top right of VS Code select the poetry environment (should be something like `~\AppData\local\pypoetry\Cache\virtualenvs\gnss_lib-<XXXX}\Scripts\python.exe` on Windows). Then you should be able to run the notebooks in VS Code.

Note: may need to run `poetry add jupyter notebook` in order for poetry env to be visible in list of VS Code kernels.

## Testing
To run the test suite, you can use pytest.
```
poetry run pytest tests/
```
or use the commands
```
poetry shell
python -m pytest
```
from the parent directory

## Building documentation
We are currently working with local builds of Sphinx documentation. 
Our plan is to migrate to [ReadTheDocs](https://readthedocs.org/) once 
we release the Beta version of the repository to the public.

Use the following steps to build and view the documentation locally.

If you made changes to filenames or moved files between directories,
run the following from the `docs` directory:

```./rebuild_references.sh```

If you also changed directory names:

* update `docs/conf.py` to reflect correct directory names
* update the helper tool `/docs/rebuild_references.sh`
* search the entire package files to check that all references to the
    directory have been changed

If you changed python dependencies:

* add the new dependency to the poetry dependency list with
    `poetry add package=version` or if the dependency is a
    development tool `poetry add --dev package=version`
* export update requirements.txt file for sphinx by running the
    following from the main directory:
    `poetry export -f requirements.txt --output ./docs/source/requirements.txt`

After the above, run the following commands from the `docs`
directory to update the documentation source and generate a local
HTML version:
```
make clean
make html
```

After building the html, you can open `docs/build/html/index.html` in
a browser to inspect your local copy of the documentation.
