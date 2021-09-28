Install
=======

Standard Installation
---------------------

Install dependencies with pip:

.. code-block:: bash

    pip3 install -r requirements.txt

Clone the GitHub repository:

.. code-block:: bash

    git clone https://github.com/Stanford-NavLab/gnss_lib_py.git

.. _developer install:

Developer Installation
----------------------

This project is being developed using :code:`pyenv` and :code:`poetry` 
for python version and environment control respectively. 
Additionally,our code has been written in and tested for Ubuntu 18 and 
20 (standalone) and Ubuntu 20 on WSL2.

1. Follow the installation section `here for Windows
   <https://pypi.org/project/pyenv-win/>`_ or 
   `here for Linux/WSL2 <https://github.com/pyenv/pyenv#installation>`_.
   After completing steps 1-3 for Windows, or 1-5 for Linux,  
   install Python using :code:`pyenv install 3.8.9`. We currently 
   support Python version 3.8.9 and above. 
2. Finish the installation using the remaining steps in the :code:`pyenv`
   documentation. 
3. Install :code:`poetry` using the instructions 
   `here <https://python-poetry.org/docs/#installation>`_.
4. Clone our repository. Use the command `pyenv local` and verify that 
   the Python version is :code:`3.8.9+`.
5. Install dependencies using :code:`poetry install`.
6. Verify that the code is working by running tests on the code using

   .. code-block:: bash

      poetry shell
      python -m pytest

   Check the :ref:`Testing<testing>` section in the Contribution guide 
   for more details

Refer to the :ref:`Documentation<documentation>` section once you add 
code/documentation and want to build and view it locally.

Clone the GitHub repository:

.. code-block:: bash

    git clone https://github.com/Stanford-NavLab/gnss_lib_py.git
