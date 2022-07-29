.. _troubleshooting:

Troubleshooting
===============

Notebooks Won't Run with VS Code
--------------------------------

Run :code:`poetry shell` to activate the environment. Then run
:code:`code .` in order to open VS Code within the environment. At the
top right of VS Code select the poetry environment (should be something
like :code:`~\AppData\local\pypoetry\Cache\virtualenvs\gnss_lib-<XXXX}\Scripts\python.exe`
on Windows). Then you should be able to run the notebooks in VS Code.

Note: may need to run :code:`poetry add jupyter notebook` in order for
poetry env to be visible in list of VS Code kernels.

No module named *
-----------------

For example:

.. code-block:: bash

   Extension error:
   Could not import extension nbsphinx (exception: No module named 'nbsphinx')
   make: *** [Makefile:20: html] Error 2

It's possible a new dependency has been added. Try running
:code:`poetry install`.

Pandoc wasn't found
-------------------

Following error is possible when building the documentation.

.. code-block::

   Pandoc wasn't found.
   Please check that pandoc is installed:
   https://pandoc.org/installing.html

Pandoc is now a dependency for building the documentation. Check the
:ref:`Pandoc installation instructions<install_pandoc>` under the
:ref:`developer installation instructions<developer install>`.
