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

`NavData` method returns `None`
-------------------------------

If you run functions like `sort` or `interpolate` on NavData and the
function returns `None`, check the value of the `inplace` attribute.
If `inplace=True`, the input `NavData` instance has been modified and the
function will return `None`. Change to `inplace=False` to return a new
`NavData` instance with the modified data.

No module named *
-----------------

For example:

.. code-block:: bash

   Extension error:
   Could not import extension nbsphinx (exception: No module named 'nbsphinx')
   make: *** [Makefile:20: html] Error 2

It's possible a new dependency has been added. Verify that you're in the
right directory (or using the right environment if running a Jupyter
notebook) and update your environment using :code:`poetry install`.

:code:`build_docs.sh` errors
----------------------------
.. _build_errors:

When running :code:`./build_docs.sh`, it is possible to run into errors
like :code:`.build_docs.sh: command not found`.
In this case, try running :code:`bash build_docs.sh` instead.

When running :code:`./build_docs.sh`, if you get the message
:code:`rm: cannot remove './source/reference/algorithms/*'$'\r': No such file or directory`,
the line endings for :code:`build_docs.sh` might have changed to CRLF.
Change the line endings to LF and re-run the command.

Pandoc wasn't found
-------------------

The following error is possible when building the documentation.

.. code-block::

   Pandoc wasn't found.
   Please check that pandoc is installed:
   https://pandoc.org/installing.html

Pandoc is now a dependency for building the documentation. Check the
:ref:`Pandoc installation instructions<install_pandoc>` under the
:ref:`developer installation instructions<developer install>`.
