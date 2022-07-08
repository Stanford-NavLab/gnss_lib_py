.. _development:

Development Guides
==================

Here are a set of detailed guides depending on if you are a public user,
Stanford NAV Lab member, or a project maintainer.


Standard GitHub Workflow
------------------------

1. Fork `gnss_lib_py <https://github.com/Stanford-NavLab/gnss_lib_py>`__
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

6. Add your name to the `contributors list <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/CONTRIBUTORS.sh>`__.

7. When you're done making changes run all the tests with:

   .. code-block:: bash

      poetry run pytest

   See the :ref:`Testing<testing>` section for more details.

8. Verify that testing coverage has not decreased:

   .. code-block:: bash

      poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/core --cov=gnss_lib_py/parsers --cov=gnss_lib_py/utils --cov-report=xml
      poetry run coverage report

   See the :ref:`Coverage Report<coverage>` section for more details.

9. Commit your changes and publish your branch to GitHub:

   .. code-block:: bash

      git add -A
      git commit -m "<describe changes in this commit>"
      git push origin your-name/name-of-your-bugfix-or-feature

10. Submit a pull request through the GitHub website.

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

5. Add your name to the `contributors list <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/CONTRIBUTORS.sh>`__.

6. When you're done making changes run all the tests with:

   .. code-block:: bash

      poetry run pytest

   See the :ref:`Testing<testing>` section for more details.

7. Verify that testing coverage has not decreased:

   .. code-block:: bash

      poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/core --cov=gnss_lib_py/parsers --cov=gnss_lib_py/utils --cov-report=xml
      poetry run coverage report

   See the :ref:`Coverage Report<coverage>` section for more details.

8. When you're ready to commit changes follow the steps below to
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

9. Submit a pull request through the GitHub website and request as a
   step in the pull request that either Ashwin or Derek review your
   code.

Pull Request Review Workflow
----------------------------

1. Change to the branch in review:

.. code-block :: bash

   git checkout their-name/name-of-the-bugfix-or-feature

2. Update your local :code:`poetry` environment to include any
   potentially new dependencies added to poetry:

.. code-block :: bash

   poetry install

3. Review the changes and added code. Look for common sense errors,
   violated conventions or places where a better implementation is
   possible. If doing an in-depth review of an algorithm and related
   tests, verify the correctness of the math and that the tests make
   valid assumptions.

3. Verify that documentation is complete and updated if necessary. See
   the :ref:`Documentation<documentation>` section for more details on
   what to check.

4. Verify that all tests run on your system:

   .. code-block:: bash

      poetry run pytest

   See the :ref:`Testing<testing>` section for more details.

5. Verify that testing coverage has not decreased:

   .. code-block:: bash

      poetry run pytest --cov=gnss_lib_py/algorithms --cov=gnss_lib_py/core --cov=gnss_lib_py/parsers --cov=gnss_lib_py/utils --cov-report=xml
      poetry run coverage report

   See the :ref:`Coverage Report<coverage>` section for more details.

6. Submit your approval or any comments on GitHub.
