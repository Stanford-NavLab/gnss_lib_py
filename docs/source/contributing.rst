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

5. Make changes locally and document them appropriately. See the
   :ref:`Documentation<documentation>` section for more details.

6. When you're done making changes run all the tests with:

   .. code-block:: bash

      poetry run pytest tests/

   See the :ref:`Testing<testing>` section for more details.

7. Verify that testing coverage has not decreased:

   .. code-block:: bash

      poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/core --cov=gnss_lib_py/io --cov=gnss_lib_py/utils --cov-report=xml
      poetry run coverage report

   See the :ref:`Coverage Report<coverage>` section for more details.

8. Commit your changes and publish your branch to GitHub:

   .. code-block:: bash

      git add -A
      git commit -m "<describe changes in this commit>"
      git push origin your-name/name-of-your-bugfix-or-feature

9. Submit a pull request through the GitHub website.

NAVLab GitHub Workflow
++++++++++++++++++++++

1. Follow the :ref:`developer install instructions<developer install>`
to install pyenv, poetry, python dependencies, and clone the repository.

2. Update your local :code:`poetry` environment to include all packages
   being used by using :code:`poetry install`

3. Create a local branch:

    .. code-block:: bash

       git checkout -b your-name/name-of-your-bugfix-or-feature


4. Make changes locally and document them appropriately. See the
   :ref:`Documentation<documentation>` section for more details.

5. When you're done making changes run all the tests with:

   .. code-block:: bash

      poetry run pytest tests/

   See the :ref:`Testing<testing>` section for more details.

6. Verify that testing coverage has not decreased:

   .. code-block:: bash

      poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/core --cov=gnss_lib_py/io --cov=gnss_lib_py/utils --cov-report=xml
      poetry run coverage report

   See the :ref:`Coverage Report<coverage>` section for more details.

7. When you're ready to commit changes follow the steps below to
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

8. Submit a pull request through the GitHub website and request as a
step in the pull request that either Ashwin or Derek review your
code.

Pull Request Review Workflow
++++++++++++++++++++++++++++

1. Change to the branch in review:

.. code-block :: bash

   git checkout their-name/name-of-the-bugfix-or-feature

2. Update your local :code:`poetry` environment to include any
   potentially new dependencies added to poetry:

.. code-block :: bash

   poetry install

3. Verify that documentation is complete and updated if necessary. See
   the :ref:`Documentation<documentation>` section for more details on
   what to check.

4. Verify that all tests run on your system:

   .. code-block:: bash

      poetry run pytest tests/

   See the :ref:`Testing<testing>` section for more details.

5. Verify that testing coverage has not decreased:

   .. code-block:: bash

      poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/core --cov=gnss_lib_py/io --cov=gnss_lib_py/utils --cov-report=xml
      poetry run coverage report

   See the :ref:`Coverage Report<coverage>` section for more details.

6. Submit your approval or any comments on GitHub.

Package Architecture
++++++++++++++++++++

The gnss_lib_py package is broadly divided into the following sections.
Please choose the most appropriate location based on the descriptions
below for new features or functionality.

    * algorithms: This directory contains localization algorithms.
    * core: This directory contains functionality that is commonly used
      to deal with GNSS measurements.
    * io: This directory contains functions to read and process various
      GNSS data/file types.
    * utils: This directory contains visualization functions and other
      code that is non-critical to the most common GNSS use cases.

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

         poetry run pytest tests/

      Alternatively, to run tests without spawning a poetry shell, from the parent directory, run

      .. code-block:: bash

        poetry run pytest tests/

    * Within each test file, name each individual test function as
      `test_funcname`.
    * While writing your tests, you might need to use certain fixed
      objects (tuples, strings etc.). Use :code:`@pytest.fixture` to
      define such objects. Fixtures can be composed to create a fixture of a fixture.
    * As far as possible, use fixtures to get fixed
      inputs to the function and use functions that don't require an
      input or return an output.
    * When creating plots in a test, ensure that all plots are saved for
      checking later on. Plots that are created must be closed using
      :code:`plt.close()` before the tests stop running.

.. _coverage:

Coverage Report
+++++++++++++++
In general, you should not submit new functionality without also
providing corresponding tests for the code. Testing coverage reports
are indicated at the top of the GitHub repository and can be generated
locally with the following commands:

.. code-block:: bash

   poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/core --cov=gnss_lib_py/io --cov=gnss_lib_py/utils --cov-report=xml
   poetry run coverage report

The total percentage of code covered (bottom right percentage) is the
main number of priority.

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

To reference textbooks/papers in the docstrings, create a new section
titled References and include the reference as shown below in the
docstring. (Remove the block comment flag when inserting in already
written docstrings)

.. code-block :: python

    """
    References
    ----------
    .. [1] Fu, Guoyu Michael, Mohammed Khider, and Frank van Diggelen.
        "Android Raw GNSS Measurement Datasets for Precise Positioning."
        Proceedings of the 33rd International Technical Meeting of the
        Satellite Division of The Institute of Navigation (ION GNSS+
        2020). 2020.
    """

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
    * Variable and class names should be readable and follow the general
      convention of :code:`generalcategory_subcategory`, eg.
      :code:`meas_gnss` and :code:`meas_lidar`

File Header
^^^^^^^^^^^
You should begin with formatting similar to the example below following
the PEP 8 style guide for
`imports <https://www.python.org/dev/peps/pep-0008/#imports>`__ and
author and date inclusions
(`dunders <https://www.python.org/dev/peps/pep-0008/#module-level-dunder-names>`__).

.. code-block :: python

   """ Short description of the file contents.

   Lengthier description of the file contents that may span across
   multiple lines if necessary. All descriptions should start with a
   capital letter and end with a period. There should also be a
   blank line before the close of the docstring.

   """

   __authors__ = "Firstname Lastname, Firstname Lastname"
   __date__ = "DD Mmm YYYY"

   import os # import statements from the standard Python library
   import sys

   import numpy as np # a blank line and then third-party imports
   import scipy as sp

   from core.constants import CoordConsts # a blank line then gnss_lib_py imports

Citations
^^^^^^^^^
Citations should be added on a function by function basis.

TODO: ADD GUIDE FOR HOW TO CITE BASED ON AMOUNT OF CHANGED CODE FROM
SOURCE

Miscellaneous Notes
^^^^^^^^^^^^^^^^^^^
    * MATLAB is correctly written with all capital letters.
    * GitHub is correctly written with the G & H capitalized.
    * Vectors (lists, np.ndarrays, etc.) for a single time instance
      should be column vectors.
    * Collections of vectors should be 2D structures with each column
      representing the value of the vector for a particular time. In
      this convention, time varies across columns while physical
      quantities vary across rows.
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

If you changed any directory names in the repository:

    * update :code:`docs/conf.py` to reflect correct directory names
    * update the helper tool :code:`/build_docs.sh`
    * search the entire package files to check that all references to the
      directory have been changed

If you changed python dependencies:

    * add the new dependency to the poetry dependency list with
      :code:`poetry add package=version` or if the dependency is a
      development tool :code:`poetry add --dev package=version`
    * export update requirements.txt file for sphinx by running the
      following from the main directory:
      :code:`poetry export -f requirements.txt --output ./docs/source/requirements.txt`

After the above, you can run the helper tool from the main directory
that will automatically rebuild references and build a local HTML copy
of the documentation:

    .. code-block:: bash

       ./build_docs.sh

After building the html, you can open :code:`docs/build/html/index.html` in
a browser to inspect your local copy.

References
----------
Contribution guide based off of the `AdaptiveStressTestingToolbox
<https://ast-toolbox.readthedocs.io/en/latest/contributing.html>`_.
