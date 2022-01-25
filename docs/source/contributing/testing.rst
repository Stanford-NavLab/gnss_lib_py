Testing and Coverage Reports
============================

This page details our strategy for testing and coverage reports.

.. _testing:

Testing
-------

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

         poetry run pytest

      Alternatively, to run tests without spawning a poetry shell, from the parent directory, run

      .. code-block:: bash

        poetry run pytest

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

Coverage Reports
----------------
In general, you should not submit new functionality without also
providing corresponding tests for the code. Visual testing coverage
reports can be found at the top of the GitHub repository. Similar
reports can be generated locally with the following commands:

.. code-block:: bash

   poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/core --cov=gnss_lib_py/parsers --cov=gnss_lib_py/utils --cov-report=xml
   poetry run coverage report

The total percentage of code covered (bottom right percentage) is the
main number of priority.
