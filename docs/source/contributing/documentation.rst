.. _documentation:

Documentation and Style Guide
=============================

This page serves as a guide on how to add to the documentation for
developers.
If you are using, :code:`gnss_lib_py`, documentation is hosted
:ref:`here <mainpage>`.
If you are looking for functional level reference documentation, check
the :ref:`reference page <reference>`.
For tutorials on how to use :code:`gnss_lib_py`, refer to the
:ref:`tutorials page <tutorials>`.

We use `numpy docstrings
<https://numpydoc.readthedocs.io/en/latest/format.html>`__
for all documentation within this package. You can see some example
numpy docstrings `here <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`__.
In addition to class and function docstrings, any line of code
whose behaviour or purpose is not obvious, should be have a brief explanation
near the code using comments.

Additional documentation guidelines
-----------------------------------

Referring textbooks or papers
+++++++++++++++++++++++++++++

To reference textbooks/papers in the docstrings, create a new section
titled References and include the reference as shown below in the
docstring. (Remove the block comment flag when inserting in already
written docstrings)

The same reference number cannot be used across multiple methods or classes
in the same file. If you want to use the same reference in multiple methods,
you will have to update the reference number each time that the reference
is added.

.. code-block:: python

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
++++++++++++++++++++++

Following the numpy docstring formatting, the type of all parameters and
returns should be indicated. Common parameter/return types include the
following:

    * :code:`bool`
    * :code:`int`
    * :code:`float`
    * :code:`list` (include type of each list element in the description)
    * :code:`dict` (include key type and value type in description)
    * :code:`np.ndarray` (include shape in the description). Where possible,
      single axis arrays should be rows and time/different measurements
      should be across the columns
    * :code:`pd.DataFrame`
    * :code:`gnss_lib_py.navdata.navdata.NavData`

If methods can accept/return multiple types of values, combine them in the
parameter/return type :code:`parameter_name : list/np.ndarray` and provide
additional details in the description.

Similarly, if a method accepts/returns a particular combination of types,
mention them in the type like :code:`parameter_name : tuple of np.ndarray`
and provide details in the description.

PEP 8 Style Guide
-----------------
We also follow the `PEP 8 Python Style Guide
<https://www.python.org/dev/peps/pep-0008/>`__.
Highlights from PEP 8 include:

    * Class names should be in CamelCase
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
-----------
You should begin with formatting similar to the example below following
the PEP 8 style guide for
`imports <https://www.python.org/dev/peps/pep-0008/#imports>`__ and
author and date inclusions
(`dunders <https://www.python.org/dev/peps/pep-0008/#module-level-dunder-names>`__).

.. code-block:: python

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

   import gnss_lib_py.utils.constants as consts # a blank line then gnss_lib_py imports

Citations
---------
Citations should be added on a function by function basis.

If a function is built on the implementation from another repository,
include the license and attribution as required by the original author.

Use the :code:`Notes` and :code:`References` sections in the docstring
to ensure proper citations, links, and explanations are provided.

Miscellaneous Style Notes
-------------------------
    * Vectors (lists, np.ndarrays, etc.) for a single time/measurement instance
      should be column vectors.
    * Collections of vectors should be 2D structures with each column
      representing the value of the vector for a particular time. In
      this convention, time varies across columns while physical
      quantities vary across rows.
    * When there are multiple measurements per time instance, there should
      be a row specifying the time of each measurement. Each measurement
      instance should be a column vector. Different types of information
      should be rows. Eg., for 10 measurements across 4 time instances,
      containing 6 types of information, the shape of the data should be
      :code:`(6, 40)`. Ideally, the first 10 columns should correspond to
      the same time instance, the next 10 for the next time instance, and
      so on.
    * Assert errors and tell the user what caused that particular error.
      For example, if a column vector is passed instead of a row vector,
      the assertion error message should say that a row vector was
      expected. We maintain functions in :code:`utils/*` that might be
      useful for performing such checks. Please check if an existing
      function performs the desired task before adding new functions.
    * Write units in brackets in comments and docstrings. For example,
      [m] for values of distance/position and [m/s] for velocities.


Adding to Documentation Pages
-----------------------------

If you find that documentation added to the code is not enough for your
intended use, you can add a page to the Sphinx documentation.

Use the `RST Cheat Sheet
<https://sphinx-tutorial.readthedocs.io/cheatsheet/>`__ from the Sphinx
documentation for any syntax queries.

Building Documentation
----------------------

Updating Documentation Configuration
++++++++++++++++++++++++++++++++++++

If you changed any directory names in the repository:

    * update :code:`docs/conf.py` to reflect correct directory names
    * update the helper tool :code:`build_docs.sh`
    * search the entire package files to check that all references to the
      directory have been changed

If you wish to add python dependencies:

    * add the new dependency to the poetry dependency list with
      :code:`poetry add package=version` or if the dependency is a
      development tool :code:`poetry add package=version --group dev`

If you wish to remove python dependencies, use :code:`poetry remove package`.

Building Documentation Locally
++++++++++++++++++++++++++++++

If you're using :code:`poetry`, after the above, you can run the helper
tool from the main directory that will automatically rebuild references
and build a local HTML copy of the documentation:

    .. code-block:: bash

       ./build_docs.sh

After building the html, you can open :code:`docs/build/html/index.html` in
a browser to view your local copy.

If you encounter errors while using the :code:`build_docs.sh` tool, refer
to previously documented solutions in the
:ref:`troubleshooting page <build_errors>`.

Building Documentation on ReadTheDocs
+++++++++++++++++++++++++++++++++++++

Documentation on readthedocs.com is automatically built when a new pull
request is submitted through our GitHub Actions. Check the corresponding
branch of the documentation after you have submitted a pull request
`here <https://readthedocs.org/projects/gnss-lib-py/versions/>`.
