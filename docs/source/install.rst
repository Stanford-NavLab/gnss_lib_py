.. _install:

Install
=======

Prerequisites
-------------

| **Python:** >=3.8, <3.11
| **Operating System:** Ubuntu, Windows, MacOS

All :code:`gnss_lib_py` classes and methods are tested in Python 3.8
and 3.10 in the latest Ubuntu, MacOS and Windows versions.
:code:`gnss_lib_py` is developed in Python 3.8.9 in Ubuntu 20/22 and
Ubuntu 20 for WSL2.

Standard Installation
---------------------

1. :code:`gnss_lib_py` is available through :code:`pip` installation
   with:

   .. code-block:: bash

      pip install gnss-lib-py

Editable Installation
---------------------

1. Clone the GitHub repository:

   .. code-block:: bash

      git clone https://github.com/Stanford-NavLab/gnss_lib_py.git

2. Install dependencies with pip:

   .. code-block:: bash

       pip3 install -r requirements.txt

3. Update pip version.

   a. For Linux:

      .. code-block:: bash

         pip install -U pip

   b. For Windows:

      .. code-block:: bash

          python -m pip install -U pip

4. Install :code:`gnss_lib_py` locally from directory containing :code:`setup.py`

   .. code-block:: bash

      pip install -e .

5. Verify installation by running :code:`pytest`.
   A successful installation will be indicated by all tests passing.

   .. code-block:: bash

      pytest

.. _developer install:

Developer Installation
----------------------

This project is being developed using :code:`pyenv` and :code:`poetry`
for python version and environment control respectively.

Ubuntu/WSL2
+++++++++++

1. Install :code:`pyenv` using the installation instructions
   `here <https://github.com/pyenv/pyenv#installation>`__. The steps are
   briefly summarized below:

   a. Install the `Python build dependencies <https://github.com/pyenv/pyenv/wiki#suggested-build-environment>`__.

   b. Either use the `automatic installer <https://github.com/pyenv/pyenv-installer>`__
      or the `Basic GitHub Checkout <https://github.com/pyenv/pyenv#basic-github-checkout>`__.

   c. In either case, you will need to configure your shell's
      environment variables for :code:`pyenv` as indicated in the install
      instructions. For example, for :code:`bash`, you can add the
      following lines to the end of your :code:`.bashrc`

      .. code-block:: bash

         export PATH="$HOME/.pyenv/bin:$PATH"
         eval "$(pyenv init --path)"
         eval "$(pyenv virtualenv-init -)"

2. Install Python 3.8.9 or above with :code:`pyenv`. For example,
   :code:`pyenv install 3.8.9`.

3. Clone the :code:`gnss_lib_py` repository.

4. Inside the :code:`gnss_lib_py` run :code:`pyenv local 3.8.9` (switching
   out with the version of Python you installed in the previous step
   if different than 3.8.9) to set the Python version that code in the
   repository will run.

5. Install :code:`poetry` using the instructions
   `here <https://python-poetry.org/docs/master/#installation>`__.

6. Install Python dependencies using :code:`poetry install`.

.. _install_pandoc:

7. Install pandoc to be able to build documentation. See details
   `here <https://pandoc.org/installing.html>`__.

   a. For Linux :code:`sudo apt install pandoc`

   b. For Windows :code:`choco install pandoc`

   c. For MacOS :code:`brew install pandoc`


8. Verify that the code is working by running tests on the code using

   .. code-block:: bash

      poetry run pytest

   Check the :ref:`Testing<testing>` section in the Contribution guide
   for more details

9. Verify that the documentation is building locally using

   .. code-block:: bash

      ./build_docs.sh

Windows
+++++++

1. Currently, full support is not offered for Windows, but :code:`pyenv`
   can be installed following instructions
   `here <https://pypi.org/project/pyenv-win/>`__.

2. The workflow for installing :code:`poetry` and :code:`gnss_lib_py` is
   similar once :code:`pyenv` has been set up.


Refer to the :ref:`Documentation<documentation>` section once you add
code/documentation and want to build and view the documentation locally.
