.. _reference:

Reference
=========

Package Architecture
--------------------

The gnss_lib_py package is broadly divided into the following sections.
Please choose the most appropriate location based on the descriptions
below for new features or functionality.

    * :code:`algorithms` : This directory contains localization algorithms.
    * :code:`core` : This directory contains functionality that is commonly used
      to deal with GNSS measurements.
    * :code:`parsers` : This directory contains functions to read and process various
      GNSS data/file types.
    * :code:`utils` : This directory contains visualization functions and other
      code that is non-critical to the most common GNSS use cases.

Details about Measurement Class
-------------------------------
Reasons that our Measurement Class is awesome

    * Ashwin wrote it.


Standard Naming Conventions
---------------------------

In large part our conventions follow from `Google's naming pattern <https://www.kaggle.com/c/google-smartphone-decimeter-challenge>`_


Those standard names are as follows:

  * :code:`trace_name`
  * :code:`rx_name`
  * :code:`gps_week` : (int) GPS week info...
  * :code:`gps_tow` : (float) time of receiving signal as measured by
    the receiver in seconds since start of GPS week
  * :code:`gnss_id` : (int) GNSS identification number using
    the following mapping

      *  0 : UNKNOWN
      *  1 : GPS
      *  2 : SBAS
      *  3 : GLONASS
      *  4 : QZSS
      *  5 : BEIDOU
      *  6 : GALILEO
      *  7 : IRNSS

  * :code:`sv_id` : (int) satellite vehicle identification number
  * :code:`signal_type`
  * :code:`tx_sv_tow` (float) measured signal transmission time as
    sent by the space vehicle/satellite and in seconds since the start
    of the gps week.
  * :code:`x_sat_m` : (float) satellite ECEF x position in meters at best
    estimated true signal transmission time.
  * :code:`y_sat_m` : (float) satellite ECEF x position in meters at best
    estimated true signal transmission time.
  * :code:`z_sat_m` : (float) satellite ECEF x position in meters at best
    estimated true signal transmission time.
  * :code:`vx_sat_mps`
  * :code:`vy_sat_mps`
  * :code:`vz_sat_mps`
  * :code:`b_sat_m`
  * :code:`b_dot_sat_mps`
  * :code:`raw_pr_m`
  * :code:`corr_pr_m`
  * :code:`raw_pr_sigma_m`
  * :code:`intersignal_bias_m`
  * :code:`iono_delay_m`
  * :code:`tropo_delay_m`
  * :code:`cn0_dbhz` : (float) carrier-to-noise density in dB-Hz
  * :code:`accumulated_delta_range_m` :
  * :code:`accumulated_delta_range_sigma_m` :



Module Level Function References
--------------------------------
All functions and classes are fully documented in the linked
documentation below.

  .. toctree::
     :maxdepth: 2

     algorithms/modules
     core/modules
     parsers/modules
     utils/modules


Additional Indices
------------------

* :ref:`genindex`
* :ref:`modindex`
