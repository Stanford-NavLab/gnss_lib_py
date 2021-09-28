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
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Feature requests and feedback
-----------------------------

The best way to send feedback is to file an issue on
`GitHub <https://github.com/Stanford-NavLab/gnss_lib_py/issues>`_.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions are welcome :)

Development
-----------

Standard GitHub Workflow
++++++++++++++++++++++++

1. Fork `gnss_lib_py <https://github.com/Stanford-NavLab/gnss_lib_py>`_
   (look for the "Fork" button).

2. Clone your fork locally::

    git clone https://github.com/<your username>/gnss_lib_py

3. Follow the :ref:`developer install instructions<developer install>`
to install pyenv, poetry, and the python dependencies:

.. code-block:: bash

    git checkout -b your-name/name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. When you're done making changes run all the tests with::

    poetry run pytest tests/

See the Testing and Documenting sections for more details.

5. Commit your changes and push your branch to GitHub:

.. code-block:: bash

    git add -A
    git commit -m "<describe changes in this commit>"
    git push origin your-name/name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website.

NAVLab GitHub Workflow
++++++++++++++++++++++

1. Follow the :ref:`developer install instructions<developer install>`
to install pyenv, poetry, python dependencies, and clone repo::

    git checkout -b your-name/name-of-your-bugfix-or-feature

2. Make changes and document them appropriately.

3. When you're done making changes run all the tests with::

    poetry run pytest tests/

4. When you're ready to commit changes follow the steps below to
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

5. Submit a pull request through the GitHub website and request as a
step in the pull request that either Ashwin or Derek to review your
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

Testing
+++++++

TODO: TESTING EXPLANATIONS

Documentation
+++++++++++++

We use `numpy docstrings
<https://numpydoc.readthedocs.io/en/latest/format.html>`_
for all documentation within this package. You can see some example
numpy docstrings `here <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`_.

Building Documentation
++++++++++++++++++++++

If you made changes to filenames or moved files between directories,
run the following from the `docs` directory::

    ./rebuild_references.sh

If you also changed directory names, you'll have to change the
`/docs/rebuild_references.sh` helper tool accordingly.

Run the following commands from the `docs` directory to update the
documentation source and generate a local HTML version::

   make clean
   make html

After building the html, you can open `docs/build/html/index.html` in
a browser to inspect your local copy.

References
----------
Contribution guide based off of the `AdaptiveStressTestingToolbox
<https://ast-toolbox.readthedocs.io/en/latest/contributing.html>`_.
