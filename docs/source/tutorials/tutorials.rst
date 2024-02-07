.. _tutorials:

Tutorials
=========


This library is meant to be used to easily interact with standard
GNSS datasets and measurement types to run standard baseline algorithms.

The tutorials below show you how to interact with our standard
:code:`NavData` class and how to run standard baselines all with only a
few lines of code.

NavData Tutorials
-----------------

These tutorials show how to interact with our standard :code:`NavData`
class and its corresponding operations.

.. toctree::
   :maxdepth: 2

   navdata/tutorials_navdata_notebook
   navdata/tutorials_operations_notebook


Parser Tutorials
----------------

These tutorials explain details about existing parsers and how to create
a new parser if necessary.

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
   parsers/tutorials_new_parsers_notebook

Algorithm Tutorials
-------------------

These tutorials demonstrate existing algorithms implemented in ``gnss_lib_py``.

.. toctree::
   :maxdepth: 2

   algorithms/tutorials_fde_notebook
   algorithms/tutorials_residuals_notebook
   algorithms/tutorials_gnss_filters_notebook
   algorithms/tutorials_snapshot_notebook


Utility Tutorials
-----------------

These tutorials illustrate a few of the most common utility functions
available in the :code:`utils` directory.

.. toctree::
   :maxdepth: 2

   utils/tutorials_constants_notebook
   utils/tutorials_coordinates_notebook
   utils/tutorials_ephemeris_downloader_notebook
   utils/tutorials_file_operations_notebook
   utils/tutorials_filters_notebook
   utils/tutorials_gnss_models_notebook
   utils/tutorials_sv_models_notebook
   utils/tutorials_time_conversions_notebook

Visualization Tutorials
-----------------------

These tutorials illustrate a few of the most common plotting functions
available in the :code:`visualizations` directory.

.. toctree::
   :maxdepth: 2

   visualizations/tutorials_plot_metric_notebook
   visualizations/tutorials_plot_map_notebook
   visualizations/tutorials_plot_skyplot_notebook
   visualizations/tutorials_style_notebook
