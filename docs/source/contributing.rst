Contributing
============

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

Bug reports
-----------

To report a bug, please submit an issue on
`GitHub <https://github.com/Stanford-NavLab/gnss_lib_py/issues>`_.
Please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in
      troubleshooting.
    * A minimal working example to reproduce the bug.

Feature requests and feedback
-----------------------------

The best way to send feedback is to file an issue on
`GitHub <https://github.com/Stanford-NavLab/gnss_lib_py/issues>`_.

If you are proposing a feature:

    * Explain in detail the intended feature, its purpose and how it would work.
    * Keep the scope as narrow as possible, to make it easier to
      implement.
    * Remember that this is a volunteer-driven project, and that code
      contributions are welcome :)

Development
-----------

Standard GitHub Workflow
++++++++++++++++++++++++

1. Fork `gnss_lib_py <https://github.com/Stanford-NavLab/gnss_lib_py>`_
   (look for the "Fork" button).

2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/<your username>/gnss_lib_py

3. Follow the :ref:`developer install instructions<developer install>`
to install pyenv, poetry, and the python dependencies.

4. Create a local branch:

   .. code-block:: bash

      git checkout -b your-name/name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. When you're done making changes run all the tests with:

   .. code-block:: bash

      poetry run pytest tests/

See the :ref:`Testing<testing>` and :ref:`Documenting<documentation>` sections for more details.

5. Commit your changes and publish your branch to GitHub:

   .. code-block:: bash

      git add -A
      git commit -m "<describe changes in this commit>"
      git push origin your-name/name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website.

NAVLab GitHub Workflow
++++++++++++++++++++++

1. Follow the :ref:`developer install instructions<developer install>`
to install pyenv, poetry, python dependencies, and clone the repository:

    .. code-block:: bash

       git checkout -b your-name/name-of-your-bugfix-or-feature


2. Update your local :code:`poetry` environment to include all packages 
   being used by using :code:`poetry install`

3. Make changes and document them appropriately.

4. When you're done making changes run all the tests with:

   .. code-block:: bash

      poetry shell
      python -m pytest

5. When you're ready to commit changes follow the steps below to
minimize unnecessary merging. This is especially important if multiple
people are working on the same branch. If you pull new changes, then
repeat the tests above to double check that everything is still working
as expected.

    .. code-block:: bash

        git stash
        git pull
        git stash apply
        git add <files to add to commit>
        git commit -m "<describe changes in this commit>"
        git push origin your-name/name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website and request as a
step in the pull request that either Ashwin or Derek review your
code.

Package Architecture
++++++++++++++++++++

The gnss_lib_py package is broadly divided into the following sections.
Please choose the most appropriate location based on the descriptions
below for new features or functionality.

    * algorithms: This directory contains TODO: DESCRIPTION
    * core: This directory contains TODO: DESCRIPTION
    * io: This directory contains TODO: DESCRIPTION
    * utils: This directory contains TODO: DESCRIPTION

.. _testing:

Testing
+++++++

TODO: UPDATE TESTING EXPLANATIONS

    * Tests are placed outside the source code in the tests directory.
    * Currently, the structure of the tests directory is expected to
      mirror the source directory.
    * For each file in the source directory, place the corresponding
      test, named as :code:`test_srcfname.py`, in the folder corresponding
      to the structure in :code:`gnss_lib_py`.
    * Use pytest to write and implement the tests. To run previously
      written tests, go to the parent directory and run

      .. code-block:: bash

         poetry shell
         python -m pytest

      Alternatively, to run tests without spawning a poetry shell, from the parent directory, run

      .. code-block::bash

        poetry run pytest tests/

    * Within each test file, name each individual test function as
      `test_funcname`. 
    * While writing your tests, you might need to use certain fixed
      objects (tuples, strings etc.). Use :code:`@pytest.fixture` to
      define such objects. Fixtures can be composed to create a fixture of a fixture.
    * As far as possible, use fixtures to get fixed
      inputs to the function and use functions that don't require an
      input or return an output.

.. _documentation:

Documentation
+++++++++++++

We use `numpy docstrings
<https://numpydoc.readthedocs.io/en/latest/format.html>`_
for all documentation within this package. You can see some example
numpy docstrings `here <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`_.
In addition to class and function docstrings, any section of code that
whose function is not blatantly obvious, should be independently
commented.

Parameter/Return Types
^^^^^^^^^^^^^^^^^^^^^^
Following the numpy docstring formatting, the type of all parameters and
returns should be indicated. Common parameter/return types include the
following:

    * :code:`bool`
    * :code:`int`
    * :code:`float`
    * :code:`list` (include shape in the description)
    * :code:`dict` (include key type and value type in description)
    * :code:`np.ndarray` (include shape in the description). Where possible,
      single axis arrays should be rows and time should be across
      the columns
    * :code:`pd.DataFrame`

PEP 8 Style Guide
^^^^^^^^^^^^^^^^^
We also follow the `PEP 8 Python Style Guide
<https://www.python.org/dev/peps/pep-0008/>`_. Highlights from PEP 8
include:

    * Classes names should be in CamelCase
    * Function names should be in snake_case (lowercase with words
      separated by underscores)
    * Variable names are also in snake_case (lowercase with words
      separated by underscores)
    * Constants are usually defined on a module level and written in all
      capital letters with underscores separating words. Examples
      include MAX_OVERFLOW and TOTAL
    * mixedCase is allowed only in contexts where that's already the
      prevailing style (e.g. threading.py), to retain backwards
      compatibility
    * Line lengths should generally be limited to 72 characters

File Header
^^^^^^^^^^^
Use the following header for each file:

    ::

        ########################################################################
        # Author(s):    F. Lastname
        # Date:         DD Mmm YYYY
        # Desc:         Short helpful description
        ########################################################################

Citations
^^^^^^^^^
Citations should be added on a function by function basis.

TODO: ADD GUIDE FOR HOW TO CITE BASED ON AMOUNT OF CHANGED CODE FROM
SOURCE

Miscellaneous Notes
^^^^^^^^^^^^^^^^^^^
    * MATLAB is correctly written with all capital letters.
    * GitHub is correctly written with the G & H capitalized.
    * Vectors (lists, np.ndarrays, etc.) should be rows and time should
      be across columns.
    * Assert errors and tell the user what caused that particular error.
      For example, if a column vector is passed instead of a row vector,
      the assertion error message should say that a row vector was
      expected. We maintain functions in :code:`utils/*` that might be 
      useful for performing such checks. Please check if an existing 
      function performs the desired task before adding new functions.
    * Write units in brackets in comments and docstrings. For example,
      [m].


Adding to Documentation Pages
+++++++++++++++++++++++++++++

If you find that documentation added to the code is not enough for your 
intended use, you can add a page to the Sphinx documentation.

Use the `RST Cheat Sheet
<https://sphinx-tutorial.readthedocs.io/cheatsheet/>`_ from the Sphinx 
documentation for any syntax queries.

Building Documentation
++++++++++++++++++++++

If you made changes to filenames or moved files between directories,
run the following from the :code:`docs` directory:

    .. code-block:: bash

        ./rebuild_references.sh

If you also changed directory names:

    * update :code:`docs/conf.py` to reflect correct directory names
    * update the helper tool :code:`/docs/rebuild_references.sh`
    * search the entire package files to check that all references to the
      directory have been changed

If you changed python dependencies:

    * add the new dependency to the poetry dependency list with
      :code:`poetry add package=version` or if the dependency is a
      development tool :code:`poetry add --dev package=version`
    * export update requirements.txt file for sphinx by running the
      following from the main directory:
      :code:`poetry export -f requirements.txt --output ./docs/source/requirements.txt`

After the above, activate your poetry environment from the parent 
directory using :code:`poetry shell` and run the following commands 
from the :code:`docs` directory to update the documentation source and 
generate a local HTML version:

    .. code-block:: bash

       make clean
       make html

After building the html, you can open :code:`docs/build/html/index.html` in
a browser to inspect your local copy.

References
----------
Contribution guide based off of the `AdaptiveStressTestingToolbox
<https://ast-toolbox.readthedocs.io/en/latest/contributing.html>`_.
