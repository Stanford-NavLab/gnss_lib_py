.. gnss_lib_py documentation master file, created by
   sphinx-quickstart on Tue Jul 20 15:36:43 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gnss_lib_py
===========

.. _mainpage:

:code:`gnss_lib_py` is a modular tool for parsing, analyzing, and
visualizing Global Navigation Satellite Systems (GNSS) data.
It also provides an intuitive framework allowing users to quickly
prototype, implement, and visualize GNSS algorithms.
:code:`gnss_lib_py` is modular in the sense that multiple types of
algorithms can be easily exchanged for each other and extendable in
facilitating user-specific extensions of existing implementations.

The data parers in the :code:`parsers` directory allow for loading
GNSS data from the following sources into the
:code:`gnss_lib_py`'s unifying :code:`Measurement` class:

    * rinex
    * nmea
    * sp3
    * `Google Android Derived Dataset <https://www.kaggle.com/c/google-smartphone-decimeter-challenge>`__

The following algorithms are implemented in the :code:`algorithms`
directory and work by passing in a :code:`Measurement` class.

    * Weighted Least Squares
    * Extended Kalman Filter using only GNSS measurements
    * Calculating pseudorange residuals

The following data visualization tools are available in the
:code:`utils` directory:

    * Skyplot: showing the movement of GNSS satellites during the
      elapsed time of the provided :code:`Measurement` class.
    * Metric plotting: allows you to plot a specific array of data
      from the :code:`Measurement` class
    * Residual plotting: specifically optimized for plotting residuals.

Installation
++++++++++++
For directions on how to install the :code:`gnss_lib_py` project, please
see the :ref:`install instructions<install>`.

Tutorials
+++++++++
We have a range of tutorials on how to easily use this project. They can
all be found in the :ref:`tutorials section<tutorials>`.

Contributing
++++++++++++
If you have a bug report or would like to contribute to our repository,
please head over the :ref:`contributing page<contributing>`.

Reference
+++++++++
References on the package contents, explanation of the benefits of our
custom measurement class, and function-level documentation can all be
found on our :ref:`reference section<reference>`.

Troubleshooting
+++++++++++++++
Common troubleshooting answers can be found in :ref:`troubleshooting section<troubleshooting>`.

Attribution
+++++++++++
This project is a product of the `Stanford NAV Lab <https://navlab.stanford.edu/>`__
and currently maintained by Ashwin Kanhere and Derek Knowles. If using
this project in your own work please cite the following:

.. code-block:: bash

    @misc{knowles_2022
    author = "Derek Knowles, Ashwin Kanhere and Grace Gao",
    title = "A Modular and Extendable GNSS Python Library",
    institution = "Stanford University",
    year = "2022 [Online]",
    url = "https://github.com/Stanford-NavLab/gnss_lib_py",
    }

Additionaly, we would like to thank `all contributors <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/CONTRIBUTORS.sh>`__ to this project.


.. toctree::
   :maxdepth: 4
   :hidden:

   install
   tutorials/tutorials.rst
   reference/reference.rst
   contributing/contributing.rst
   troubleshooting
