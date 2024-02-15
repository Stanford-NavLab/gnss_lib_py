.. _tutorials:

Tutorials
=========


The goal of this library is to easily interact with standard GNSS
datasets and file types and run baseline algorithms on measurements
in these datasets/data types.

The :code:`gnss_lib_py` library is divided into submodles, as
described :ref:`here<organization>` and the tutorials are similarly
organized.

These tutorials are in interactive Jupyter notebooks and have been rendered
as part of the documentation.
You can run the code yourself by running the notebooks in the 'tutorials'
directory `here <https://github.com/Stanford-NavLab/gnss_lib_py/tree/main/notebooks/tutorials>`.
The notebooks can also be run in Google Colab without downloading the
repository by selecting the 'Open in Colab' option at the top of each
notebook.

The tutorials below show you how to load datasets, interact with our
standard :Code:`NavData` class, run baseline algorithms, generate metrics
for the resultant estimates, and visualize results and data.

All of this can be accomplished with a few lines of code and modularly.


NavData Tutorials
-----------------

These tutorials show how to initialize and use our standard :code:`NavData`
class and its corresponding operations.

.. toctree::
   :maxdepth: 2

   navdata/tutorials_navdata_notebook
   navdata/tutorials_operations_notebook

Creating a New Parser
---------------------

The parsers tutorials also contain a tutorial on how to create a new
parser that inherits from :code:`NavData` to handle new measurement types
and/or files.

.. toctree::
   :maxdepth: 2

   parsers/tutorials_new_parsers_notebook


Parser Tutorials
----------------

These tutorials explain existing parsers and how to create a new parser
if necessary.

.. toctree::
   :maxdepth: 2

   parsers/tutorials_android_notebook
   parsers/tutorials_clk_notebook
   parsers/tutorials_google_decimeter_notebook
   parsers/tutorials_nmea_notebook
   parsers/tutorials_rinex_nav_notebook
   parsers/tutorials_rinex_obs_notebook
   parsers/tutorials_smartloc_notebook
   parsers/tutorials_sp3_notebook

Algorithm Tutorials
-------------------

These tutorials demonstrate existing algorithms for state estimation
and fault detection and estimation.

.. toctree::
   :maxdepth: 2

   algorithms/tutorials_fde_notebook
   algorithms/tutorials_residuals_notebook
   algorithms/tutorials_gnss_filters_notebook
   algorithms/tutorials_snapshot_notebook


Utility Tutorials
-----------------

These tutorials illustrate some of the utility functions available in
the :code:`utils` directory.

.. toctree::
   :maxdepth: 2

   utils/tutorials_constants_notebook
   utils/tutorials_coordinates_notebook
   utils/tutorials_dop_notebook
   utils/tutorials_ephemeris_downloader_notebook
   utils/tutorials_file_operations_notebook
   utils/tutorials_filters_notebook
   utils/tutorials_gnss_models_notebook
   utils/tutorials_sv_models_notebook
   utils/tutorials_time_conversions_notebook

Visualization Tutorials
-----------------------

These tutorials illustrate most commonly used plotting functions
available in the :code:`visualizations` directory.

.. toctree::
   :maxdepth: 2

   visualizations/tutorials_plot_metric_notebook
   visualizations/tutorials_plot_map_notebook
   visualizations/tutorials_plot_skyplot_notebook
   visualizations/tutorials_style_notebook
