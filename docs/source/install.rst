.. _install:

Install
=======

Our code has been written in and tested on the following systems:

- Ubuntu 18
- Ubuntu 20
- Ubuntu 20 on WSL2

Standard Installation
---------------------

Clone the GitHub repository:

.. code-block:: bash

    git clone https://github.com/Stanford-NavLab/gnss_lib_py.git

Install dependencies with pip:

.. code-block:: bash

    pip3 install -r requirements.txt

Update pip version. For Linux:

.. code-block:: bash
   
   pip install -U pip

For Windows: 

.. code-block:: bash
   
   python -m pip install -U pip

Install :code:`gnss_lib_py` locally from directory containing :code:`setup.py`

.. code-block:: bash
   
   pip install -e .

Verify installation by running :code:`pytest`


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

7. Verify that the code is working by running tests on the code using

   .. code-block:: bash

      poetry run pytest

   Check the :ref:`Testing<testing>` section in the Contribution guide
   for more details

Windows
+++++++

1. Currently, full support is not offered for Windows, but :code:`pyenv`
   can be installed following instructions
   `here <https://pypi.org/project/pyenv-win/>`__.

2. The workflow for installing :code:`poetry` and :code:`gnss_lib_py` is
   similar once :code:`pyenv` has been set up.


Refer to the :ref:`Documentation<documentation>` section once you add
code/documentation and want to build and view the documentation locally.
