.. _development:

Development Guides
==================

Here are a set of detailed guides depending on if you are a :ref:`public user<standard_dev>`,
:ref:`Stanford NAV Lab member<navlab_dev>`, or a project maintainer
:ref:`reviewing pull requests<pr_review>`,
:ref:`creating new release branches<release_workflow>`, or
:ref:`updating dependencies and GitHub workflows<dependency_workflow>`.

.. _standard_dev:

Standard GitHub Workflow
------------------------

1. Fork `gnss_lib_py <https://github.com/Stanford-NavLab/gnss_lib_py>`__
   (look for the "Fork" button), then clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/<your username>/gnss_lib_py

2. If using poetry, follow the :ref:`developer install instructions<developer install>`
   to install pyenv, poetry, and the python dependencies. If using
   :code:`pip` or :code:`conda` for package management instead, install
   dependencies using

   .. code-block:: bash

      pip install -r requirements.txt

3. Create a local branch:

   .. code-block:: bash

      git checkout -b your-name/name-of-your-bugfix-or-feature

4. Make changes locally and document them appropriately. See the
   :ref:`Documentation<documentation>` section for more details.

   If the feature branch includes new functionality, you must also:

   * add a section in the appropriate tutorial notebook located in
     :code:`notebooks/tutorials/*`
   * if the feature is in a new file, import the file in the
     `module's import block <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/gnss_lib_py/__init__.py>`__.

   If directory level changes were made, you must also:

   * update the "Code Organization" section of the :code:`README.md`
   * update the "Code Organization" section of
     :code:`docs/source/index.rst` to match the :code:`README.md`


5. Add tests for the newly added code and ensure the new code is covered.
   See the :ref:`Testing<testing>` and :ref:`Coverage<coverage>` pages
   for more details.

6. When you're done making changes run all the tests with:

   .. code-block:: bash

      poetry run pytest

   Make sure that all tests are passing.

7. Verify that testing coverage has not decreased:

   .. code-block:: bash

      poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/navdata --cov=gnss_lib_py/parsers --cov=gnss_lib_py/utils --cov=gnss_lib_py/visualizations --cov-report=html
      poetry run coverage report

   See the :ref:`Coverage Report<coverage>` section for more details.

8. Improve code readability by linting it. Run :code:`pylint` to preview
   issues with the code:

   .. code-block:: bash

      poetry run python -m pylint path-to-file-to-lint

   Resolve issues that do not impact how you have implemented your functionality,
   such as conforming to snake case naming, removing TODOs and using suggested
   defaults.

9. Ensure that system and IDE dependent files, like those in :code:`.idea`
   folders for PyCharm and :code:`.vscode` folders for VS Code are not
   committed by updating the :code:`.gitignore` file.

10. Add your name to the `contributors list <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/CONTRIBUTORS.md>`__.

11. Commit your changes and publish your branch to GitHub:

   .. code-block:: bash

      git add -A
      git commit -m "<describe changes in this commit>"
      git push origin your-name/name-of-your-bugfix-or-feature

12. Submit a pull request through GitHub. For the base branch
    in the pull request, select the latest version release branch :code:`vX.Y.Z`
    (with the highest number of all such branches). *Do not target the*
    :code:`main` *branch in your pull request.* In the pull request,
    add a code review request for a current maintainer of the repository
    and provide a brief description of the implemented features.
    The reviewers might add comments or suggest changes to ensure
    compliance with the rest of the code.

.. _navlab_dev:

NAVLab GitHub Workflow
----------------------

1. Follow the :ref:`developer install instructions<developer install>`
   to install pyenv, poetry, python dependencies, and clone the repository.

2. Update your local :code:`poetry` environment to include all packages
   being used by using :code:`poetry install`

3. Create a local branch:

   .. code-block:: bash

      git checkout -b your-name/name-of-your-bugfix-or-feature

4. Make changes locally and document them appropriately. See the
   :ref:`Documentation<documentation>` section for more details.

   If the feature branch includes new functionality, you must also:

   * add a section in the appropriate tutorial notebook located in
     :code:`notebooks/tutorials/*`
   * if the feature is in a new file, import the file in the
     `module's import block <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/gnss_lib_py/__init__.py>`__.

   If directory level changes were made, you must also:

   * update the "Code Organization" section of the :code:`README.md`
   * update the "Code Organization" section of
     :code:`docs/source/index.rst` to match the :code:`README.md`


5. Add tests for the newly added code and ensure the new code is covered.
   See the :ref:`Testing<testing>` section for more details.

6. When you're done making changes run all the tests with:

   .. code-block:: bash

      poetry run pytest

   Make sure that all tests are passing.

7. Verify that testing coverage has not decreased:

   .. code-block:: bash

      poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/navdata --cov=gnss_lib_py/parsers --cov=gnss_lib_py/utils --cov=gnss_lib_py/visualizations --cov-report=html
      poetry run coverage report

   See the :ref:`Coverage Report<coverage>` section for more details.

8. Improve code readability by linting it. Run :code:`pylint` to preview
   issues with the code:

   .. code-block:: bash

      poetry run python -m pylint path-to-file-to-lint

   Resolve issues that do not impact how you have implemented your functionality,
   such as conforming to snake case naming, removing TODOs and using suggested
   defaults.

9. Ensure that system and IDE dependent files, like those in :code:`.idea`
   folders for PyCharm and :code:`.vscode` folders for VS Code are not
   committed by updating the :code:`.gitignore` file.

10. Add your name to the `contributors list <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/CONTRIBUTORS.md>`__.

11. When you're ready to commit changes follow the steps below to
    minimize unnecessary merging. This is especially important if
    multiple people are working on the same branch. If you pull new
    changes, then repeat the tests above to double check that everything
    is still working as expected.

   .. code-block:: bash

      git stash
      git pull
      git stash apply
      git add <files to add to commit>
      git commit -m "<describe changes in this commit>"
      git push origin your-name/name-of-your-bugfix-or-feature

12. Submit a pull request through GitHub. For the base branch
    in the pull request, select the latest version release branch :code:`vX.Y.Z`
    (with the highest number of all such branches). *Do not target the*
    :code:`main` *branch in your pull request.* In the pull request,
    add a code review request for a current maintainer of the repository.
    The reviewers might add comments to ensure compliance with the rest
    of the code.

.. _pr_review:

Pull Request Review Workflow
----------------------------

1. Change to the branch in review:

   .. code-block:: bash

      git checkout their-name/name-of-the-bugfix-or-feature

2. Update your local :code:`poetry` environment to include any
   new dependencies that might have been added to poetry:

   .. code-block:: bash

      poetry install

3. Review the changes and added code. Look for common sense errors,
   violated conventions or places where a better implementation is
   possible. If doing an in-depth review of an algorithm and related
   tests, verify the correctness of the math and that the tests make
   valid assumptions.

3. Verify that documentation is complete and updated if necessary. See
   the :ref:`Documentation<documentation>` section for more details on
   what is expected.
   If the feature branch included new functionality, the following
   should have also been updated:

   * the appropriate tutorial notebook located in
     :code:`notebooks/tutorials/*` with a simple example of the new
     functionality
   * if a new file was created, it should likely be imported in the
     `module's import block <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/gnss_lib_py/__init__.py>`__.

   If the directory structure was modified, the following should also have
   been updated:

   * the "Code Organization" section of the :code:`README.md`
   * the "Code Organization" section of
     :code:`docs/source/index.rst` to match the :code:`README.md`

4. Verify that all tests run on your system:

   .. code-block:: bash

      poetry run pytest

   See the :ref:`Testing<testing>` section for more details.

5. Verify that all status checks are passing on GitHub.
   Treat failing status checks as failed tests, doc errors or linting
   issues, depending on the corresponding GitHub Action

6. Verify that testing coverage has not decreased:

   .. code-block:: bash

      poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/parsers --cov=gnss_lib_py/utils --cov-report=html
      poetry run coverage report

   See the :ref:`Coverage Report<coverage>` section for more details.

7. Verify that the Pull Request targets the latest version release branch,
   called :code:`vX.Y.Z`. If it doesn't target this branch, change the base
   branch to the latest version release branch. If this branch
   doesn't exist, create the latest version release branch from :code:`main`
   before changing the base.

8. Submit your approval or any comments on GitHub.

.. _release_workflow:

New Package Release Workflow
----------------------------

1. Switch to the latest version release branch (with the highest number):

   .. code-block:: bash

      git checkout -b vX.Y.Z

2. Open the ``pyproject.toml`` file and under the ``[tool.poetry]``
   group change the ``version = X.Y.Z`` variable to match the new
   package version number.

3. Create a new pull request and merge to the ``main`` branch using the
   development process above.

4. Go to the `releases page <https://github.com/Stanford-NavLab/gnss_lib_py/releases>`__
   on GitHub and click the ``Draft a new release`` button on the top.
   Click ``Choose a tag`` and add a new tag named ``X.Y.Z`` matching the
   new package version number. Target the ``main`` branch. Finally,
   click the ``Publish release`` button.

5. Allow time for the release to build and then check
   `pypi <https://pypi.org/project/gnss-lib-py/>`__
   to ensure that the release was built successfully.

.. _dependency_workflow:

Updating Dependencies and GitHub Workflows
------------------------------------------

Maintaining up-to-date dependencies and GitHub Actions workflows is essential for
project security, bug fixes, and compatibility with current Python releases.
``gnss_lib_py`` uses a dual dependency structure:

* **Poetry** (``pyproject.toml`` and ``poetry.lock``) serves as the primary source of
  truth for defining and locking package dependencies and their exact versions.
* **Requirements files** (``requirements.txt`` and ``docs/source/requirements.txt``)
  are exported artifacts generated directly from ``poetry.lock``. They are consumed
  by ``pip install -r requirements.txt``, Conda setups, ReadTheDocs, and CI workflows.

This guide outlines an efficient, automated workflow for checking for updates,
refreshing dependencies, updating GitHub Actions, and verifying changes.

Workflow Overview
~~~~~~~~~~~~~~~~~

A routine dependency update consists of the following steps:

1. **Check GitHub Actions & CI environments**: Run ``python scripts/check_workflow_updates.py``
   to automatically identify outdated GitHub Actions, CI Poetry versions, or end-of-life
   Python releases.
2. **Update dependencies**: Run ``poetry update`` to resolve and lock all unpinned
   packages to their latest compatible versions.
3. **Export requirements**: Run ``poetry run poetry export`` to update both
   ``requirements.txt`` and ``docs/source/requirements.txt``.
4. **Test and verify**: Run pytest, pylint, and ``./build_docs.sh`` to ensure zero
   regressions.
5. **Submit a pull request**: Target the latest version release branch (``vX.Y.Z``).

Step 1: Checking GitHub Actions & CI Workflows Automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than manually reviewing each action in ``.github/workflows/*.yml`` and
``.readthedocs.yaml``, ``gnss_lib_py`` includes a standard-library script that
automatically checks all workflow dependencies:

.. code-block:: bash

   python scripts/check_workflow_updates.py

**What the script checks:**

* **GitHub Actions**: Inspects every ``uses: owner/repo@tag`` reference across all
  workflow files, queries the GitHub API for the latest release tag, and reports
  available updates.
* **Poetry Version in CI**: Checks the version pinned under ``snok/install-poetry@v1``
  in ``build.yml`` and ``pylint.yml`` against the latest Poetry release on PyPI.
* **Python Supported Versions**: Compares the Python versions in the CI test matrix
  and ReadTheDocs configuration against the official Python release cycle from
  `endoflife.date <https://endoflife.date/python>`__, flagging end-of-life versions or
  newly released versions that should be tested.

.. tip::
   To avoid GitHub's unauthenticated API rate limit (60 requests/hour), you can set the
   ``GITHUB_TOKEN`` environment variable before running the script:

   .. code-block:: bash

      export GITHUB_TOKEN="<your-github-token>"
      python scripts/check_workflow_updates.py

If updates are reported:

1. **Update GitHub Actions**: In ``.github/workflows/*.yml``, bump the action version tag
   (e.g., updating ``actions/checkout@v6`` to ``actions/checkout@v7``).
2. **Update Poetry in CI**: Update the ``version: X.Y.Z`` field under ``snok/install-poetry``
   in both ``.github/workflows/build.yml`` and ``.github/workflows/pylint.yml``.
3. **Update Python Matrix** (if Python versions were added or deprecated):

   * Update ``matrix.python-version`` in ``.github/workflows/build.yml``.
   * Update the single Python test version in ``.github/workflows/pip-install.yml``,
     ``.github/workflows/pylint.yml``, and ``.github/workflows/pypi-release.yml``.
   * Update ``build.tools.python`` in ``.readthedocs.yaml``.
   * Update ``python = ">=3.10, < 3.15"`` in ``pyproject.toml``.
   * Update references in ``docs/source/install.rst`` and
     ``.github/ISSUE_TEMPLATE/bug_report.md``.

Step 2: Updating Python Dependencies with Poetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do you need to run poetry show --outdated?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running :code:`poetry show --outdated` queries PyPI for every installed package to display
available newer versions (or :code:`poetry show --outdated --top-level` to view only direct
dependencies). While informative for inspecting individual packages, **it is not necessary
prior to updating**.

For routine updates and security vulnerability fixes, run :code:`poetry update` directly:

.. code-block:: bash

   poetry update

Why a Full poetry update is Recommended
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Resolves transitive vulnerabilities**: Security advisories (such as Dependabot alerts
  for ``pillow``, ``certifi``, ``urllib3``, ``nbconvert``, etc.) frequently originate from
  transitive dependencies pulled in by top-level libraries. Running :code:`poetry update`
  recomputes the dependency graph, bringing all direct and transitive dependencies up to
  their highest compatible versions without requiring you to trace and update each package
  individually.
* **Keeps lockfile clean and modern**: Ensures that sub-dependencies maintain consistent,
  tested version sets compatible with your Python version range.

Targeted Package Updates
^^^^^^^^^^^^^^^^^^^^^^^^

If you only wish to patch a specific package without modifying any other locked packages:

.. code-block:: bash

   poetry update <package-name>
   # Example:
   poetry update pillow certifi urllib3 nbconvert

.. note::
   **Upstream compatibility pins:**
   Certain packages have strict compatibility requirements that must be preserved in
   ``pyproject.toml`` unless thoroughly validated. For example, ``georinex`` requires
   specific versions of ``xarray`` (currently ``xarray = "<=2024.7.0"``) and
   ``georinex = "<=1.16.1"``. Do not remove these constraints without verifying that
   all RINEX parser unit tests pass.

Auditing for Known Vulnerabilities Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can quickly scan your environment for known CVEs using ``pip-audit``:

.. code-block:: bash

   pipx run pip-audit -r requirements.txt

Synchronizing Your Local Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After ``poetry.lock`` is updated, synchronize your active virtual environment:

.. code-block:: bash

   poetry install --sync

Step 3: Exporting and Syncing Requirements Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gnss_lib_py`` maintains two exported requirements files:

* ``./requirements.txt``: Root requirements file containing only the core runtime
  dependencies. Used for ``pip install -r requirements.txt``, Conda setups, and
  tested by ``.github/workflows/pip-install.yml``.
* ``./docs/source/requirements.txt``: Contains core dependencies plus the ``dev``
  group dependencies (Sphinx, nbsphinx, notebook, etc.). Used by ReadTheDocs
  (defined in ``.readthedocs.yaml``) and for local documentation builds.

Whenever ``poetry.lock`` changes, re-export both files:

.. code-block:: bash

   # 1. Export core requirements for pip/conda
   poetry run poetry export -f requirements.txt --output ./requirements.txt --without-hashes

   # 2. Export documentation requirements (including dev dependencies)
   poetry run poetry export -f requirements.txt --output ./docs/source/requirements.txt --with dev --without-hashes

.. tip::
   Running ``./build_docs.sh`` also performs both exports automatically before generating
   the Sphinx documentation.

Step 4: Testing and Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before committing any dependency or workflow changes, verify that all checks pass locally:

1. Run all unit tests:

   .. code-block:: bash

      poetry run pytest

2. Run code linting:

   .. code-block:: bash

      poetry run pylint $(git ls-files 'gnss_lib_py/*.py') || poetry run pylint-exit --error-fail $?

3. Verify documentation generation:

   .. code-block:: bash

      ./build_docs.sh

4. Verify clean pip installation:

   .. code-block:: bash

      python -m pip install -r requirements.txt

Step 5: Committing and Submitting Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create a dedicated branch following the lab's naming convention:

   .. code-block:: bash

      git checkout -b your-name/depends-YYYYMM

2. Stage and commit the updated configuration, lockfile, requirements, and workflow files:

   .. code-block:: bash

      git add pyproject.toml poetry.lock requirements.txt docs/source/requirements.txt .github/workflows/
      git commit -m "Update dependencies and workflows"

3. Push the branch to GitHub:

   .. code-block:: bash

      git push origin your-name/depends-YYYYMM

4. Open a pull request targeting the latest version release branch (``vX.Y.Z``).
   Ensure all GitHub Actions status checks pass.


