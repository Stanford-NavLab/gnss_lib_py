.. _development:

Development Guides
==================

Here are a set of detailed guides depending on if you are a :ref:`public user<standard_dev>`,
:ref:`Stanford NAV Lab member<navlab_dev>`, or a project maintainer
:ref:`reviewing pull requests<pr_review>` or
:ref:`creating new release branches<release_workflow>`.

.. _standard_dev:

Standard GitHub Workflow
------------------------

1. Fork `gnss_lib_py <https://github.com/Stanford-NavLab/gnss_lib_py>`__
   (look for the "Fork" button), then clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/<your username>/gnss_lib_py

2. If using poetry, follow the :ref:`developer install instructions<developer install>`
   to install pyenv, poetry, and the python dependencies. If using
   :code:`pip` or :code:`conda` for package management instead, use
   :code:`pip install -r requirements.txt` to install dependencies.

3. Create a local branch:

   .. code-block:: bash

      git checkout -b your-name/name-of-your-bugfix-or-feature

4. Make changes locally and document them appropriately. See the
   :ref:`Documentation<documentation>` section for more details.

   If the feature branch includes new functionality, you must also:

   * update the "Code Organization" section of the :code:`README.md`
   * update the "Code Organization" section of
     :code:`docs/source/index.rst` to match the :code:`README.md`
   * add a section in the appropriate tutorial notebook located in
     :code:`notebooks/tutorials/*`
   * if the feature is in a new file, import the file in the
     `module's import block <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/gnss_lib_py/__init__.py>`__.

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

11. Commit your changes and publish your branch to GitHub:

   .. code-block:: bash

      git add -A
      git commit -m "<describe changes in this commit>"
      git push origin your-name/name-of-your-bugfix-or-feature

12. Submit a pull request through GitHub. For the base branch
    in the pull request, select the latest version release branch :code:`vX.Y.Z`
    (with the highest number of all such branches). *Do not target the*
    :code:`main` *branch in your pull request.* In the pull request,
    add a code review request for a current maintainer of the repository.
    The reviewers might add comments to ensure compliance with the rest
    of the code.

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

   * update the "Code Organization" section of the :code:`README.md`
   * update the "Code Organization" section of
     :code:`docs/source/index.rst` to match the :code:`README.md`
   * add a section in the appropriate tutorial notebook located in
     :code:`notebooks/tutorials/*`
   * if the feature is in a new file, import the file in the
     `module's import block <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/gnss_lib_py/__init__.py>`__.

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

   * the "Code Organization" section of the :code:`README.md`
   * the "Code Organization" section of
     :code:`docs/source/index.rst` to match the :code:`README.md`
   * the appropriate tutorial notebook located in
     :code:`notebooks/tutorials/*` with a simple example of the new
     functionality
   * if a new file was created, it should likely be imported in the
     `module's import block <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/gnss_lib_py/__init__.py>`__.

4. Verify that all tests run on your system:

   .. code-block:: bash

      poetry run pytest

   See the :ref:`Testing<testing>` section for more details.

5. Verify that all status checks are passing on GitHub.
   Treat failing status checks as failed tests, doc errors or linting
   issues, depending on the corresponding GitHub Action

6. Verify that testing coverage has not decreased:

   .. code-block:: bash

      poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/parsers --cov=gnss_lib_py/utils --cov-report=xml
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
