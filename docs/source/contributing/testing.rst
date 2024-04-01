Testing and Coverage Reports
============================

To ensure that newly added code behaves as expected and that
modifications or new additions do not change previously written code,
:code:`gnss_lib_py` uses unit testing, implemented using :code:`pytest`.

The module :code:`pytest-cov` is also used to generate code coverage
reports.
Code coverage reports detail which lines are run and which are missed
during the execution of the tests.
Higher coverage indicates that the code was run during unit tests and we
aim to keep the code coverage above 90%.

.. _testing:

Testing
-------

The following is a description of how tests are structured and placed
in the repository.
We use :code:`pytest` to write and run tests.
A brief description of the naming convention and commonly used features
is also included below.
For more details, refer to the `pytest documentation <https://docs.pytest.org/>`__.


Running tests
+++++++++++++

  * To run tests, go to the parent directory and run

    .. code-block:: bash

       poetry run pytest

  * To run tests from a specific file run

    .. code-block:: bash

       poetry run tests/folder_name/test_file_name.py

  * To run a particular test, contained in a specific file, run

    .. code-block:: bash

       poetry run pytest tests/folder_name/test_file_name.py::test_function


Naming convention and location of tests
+++++++++++++++++++++++++++++++++++++++

  * Tests are placed outside the source code, in the :code:`tests/`
    diretory
  * The structure of the tests directory is expected to mirror the source
    directory. Eg. tests for functions in :code:`gnss_lib_py/utils/file_name.py`
    must be placed in :code:`tests/utils/test_file_name.py`
  * Following the :code:`pytest` naming convention, test functions are
    named :code:`test_function`
  * Fixtures should be named as :code:`fixture_name` where :code:`name` is
    the name of the fixture, defined using the :code:`@pytest.fixture(name="name")`
    decorator before the fixture definition. More details on expected
    uses of fixtures can be found in the `fixtures`_ section.

Conventions for writing tests
+++++++++++++++++++++++++++++

  * Most tests in :code:`gnss_lib_py` are unit tests.
    Unit tests use a known input-output pair to verify that the function
    being tested is working as expected.
    Unit tests should be used to verify nominal behaviour as well as any
    relevant edge cases

  * To verify that the function output matches the expected output, we
    recommend using the in-built :code:`assert` statement, numpy's
    testing functions, that are of the form :code:`np.testing.assert*`
    and any additional module level testing functions, like
    :code:`pd.testing.*assert*` for pandas.

  * Use _`fixtures` :code:`@pytest.fixture` to define certain fixed
    objects (tuples, strings etc.) that might be required by test functions.
    Fixtures can be named and these names can be used to access fixture
    outputs like regular variables

  * Fixtures can also be composed to create a fixture of a fixture.

  * :code:`gnss_lib_py` already uses numerous fixtures for testing. Fixtures
    that are relevant only to tests in a single file are defined in that file.
    Fixtures that are used across multiple files are defined in
    :code:`conftest.py` in the testing directory :code:`gnss_lib_py/tests/`.

  * Before defining a new fixture, check to see if the fixture exists in
    :code:`conftest.py`. If it does, you can refer to the fixture by name
    in your test function and use without having to define any additional
    methods.

  * Existing fixtures from :code:`conftest.py` also define locations of
    test files or common test data that can be used for your own fixtures.
    Please check these existing fixtures to enable quick development of
    your own.

  * If you want to test the same function beheaviour for multiple
    input-output pairs, you can use the :code:`pytest.mark.parameterize`
    function decorator before the test

  * When creating plots in a test, if you want to visually verify the
    plots,ensure that all plots are saved for checking later on.
    Plots that are created must be closed using :code:`plt.close()`
    before the tests stop running

For additional details on :code:`pytest` functionality mentioned above,
check the `pytest documentation <https://docs.pytest.org/>`__.

.. _coverage:

Coverage Reports
----------------
In general, you should not submit new functionality without also
providing corresponding tests for the code.
Visual testing coverage reports can be found at the top of the GitHub
repository.
Similar reports can be generated locally, in terminal, with the
following commands:

.. code-block:: bash

   poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/parsers --cov=gnss_lib_py/utils --cov-report=html
   poetry run coverage report

The total percentage of code covered (bottom right percentage) is the
main number of priority.

To generate the coverage report and view as a webpage, use the following
command:

.. code-block:: bash

   poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/parsers --cov=gnss_lib_py/utils --cov-report=html

The generated coverage report can be accessed from the directory :code:`htmlcov/`
