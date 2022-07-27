.. _tutorials:

Tutorials
=========


This library is meant to be used to easily interact with standard
GNSS datasets and run standard algorithms.
The code below shows how you can easily create a skyplot from a dataset
with only three function calls.

.. code-block:: python

    from gnss_lib_py.algorithms.snapshot import solve_wls
    from gnss_lib_py.parsers.android import AndroidDerived
    from gnss_lib_py.utils.visualizations import plot_skyplot

    data_path = "/google-decimeter-2021/train/2021-01-05-US-SVL-1/Pixel4XL/Pixel4XL_derived.csv"

    # convert data to Measurement class
    derived_data = AndroidDerived(data_path)

    # calculate Weighted Least Squares position estimate
    state = solve_wls(derived_data)

    # create skyplot of the satellites movement
    plot_skyplot(derived_data, state)

.. image:: ../img/skyplot.png
  :width: 400
  :alt: skyplot of GNSS movement over time



Measurement Tutorials
---------------------

Our unifying class.

TODO: simple example showing the speed up of the measurement class
and intuitive adding data to it.

.. code-block:: python

    from gnss_lib_py.parsers.navdata import NavData

    navdata = NavData()

More information on the Measurement class can be found.

.. toctree::
   :maxdepth: 1

   tutorials_measurement
   tutorials_measurement_notebook

Parser Tutorials
----------------

TODO: copy and paste from intro page

.. toctree::
   :maxdepth: 1

   tutorials_parsers

Algorithm Tutorials
-------------------

TODO: copy and paste from intro page

.. toctree::
   :maxdepth: 1

   tutorials_algorithms


Utility Tutorials
-----------------

TODO: copy and paste from intro page

.. toctree::
   :maxdepth: 1

   tutorials_utilities
