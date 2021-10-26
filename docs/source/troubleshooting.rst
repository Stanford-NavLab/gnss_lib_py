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
