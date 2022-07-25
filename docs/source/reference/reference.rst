.. _reference:

Reference
=========

Package Architecture
--------------------

The gnss_lib_py package is broadly divided into the following sections.
Please choose the most appropriate location based on the descriptions
below for new features or functionality.

    * :code:`algorithms` : This directory contains localization algorithms.
    * :code:`parsers` : This directory contains functions to read and process various
      GNSS data/file types.
    * :code:`utils` : This directory contains functionality that is
      commonly used to deal with GNSS measurements.

Details about Measurement Class
-------------------------------
Reasons that our Measurement Class is awesome

    * if initializing with np.ndarrays, rows are each data type
    * column names need to be strings
    * if initializing with a csv file without headers, use
      :code:`Measurement(csv_path = "path", header = None)`

Standard Naming Conventions
---------------------------

In large part our conventions follow from `Google's naming pattern <https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data>`_


GNSS measurement naming conventions are as follows:
  * :code:`trace_name` : (string) name for the trace
  * :code:`rx_name` : (string) name for the receiver device
  * :code:`gps_week` : (int) GPS weeks since the start of the GPS epoch
    on January 6th, 1980. The `NOAA CORS website <https://geodesy.noaa.gov/CORS/Gpscal.shtml>`__
    maintains a helpful reference calendar.
  * :code:`gps_tow` : (float) time of receiving signal as measured by
    the receiver in seconds since start of GPS week (Sunday at )
  * :code:`unix_time_millis` : (int) milliseconds that have elapsed
    since January 1, 1970 at midnight (midnight UTC) and not counting
    leapseconds.
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
  * :code:`x_sv_m` : (float) satellite ECEF x position in meters at best
    estimated true signal transmission time.
  * :code:`y_sv_m` : (float) satellite ECEF y position in meters at best
    estimated true signal transmission time.
  * :code:`z_sv_m` : (float) satellite ECEF z position in meters at best
    estimated true signal transmission time.
  * :code:`vx_sv_mps` : (float) satellite ECEF x velocity in meters per
    second at estimated true signal transmission time.
  * :code:`vy_sv_mps` : (float) satellite ECEF y velocity in meters per
    second at estimated true signal transmission time.
  * :code:`vz_sv_mps` : (float) satellite ECEF z velocity in meters per
    second at estimated true signal transmission time.
  * :code:`b_sv_m` : (float) satellite clock bias in meters.
  * :code:`b_dot_sv_mps` : (float) satellite clock bias drift in meters
    per second.
  * :code:`raw_pr_m` : (float) raw, uncorrected pseudorange in meters.
  * :code:`corr_pr_m` : (float) corrected pseudorange according to the
    formula: :code:`corr_pr_m = raw_pr_m + b_sv_m - intersignal_bias_m - iono_delay_m - tropo_delay_m`
  * :code:`raw_pr_sigma_m` : (float) uncertainty of the raw, uncorrected
    pseuodrange in meters.
  * :code:`intersignal_bias_m` : (float) inter-signal range bias in
    meters.
  * :code:`iono_delay_m` : (float) ionospheric delay in meters.
  * :code:`tropo_delay_m` : (float) tropospheric delay in meters.
  * :code:`cn0_dbhz` : (float) carrier-to-noise density in dB-Hz
  * :code:`accumulated_delta_range_m` : accumulated delta range in
    meters.
  * :code:`accumulated_delta_range_sigma_m` : uncertainty in the
    accumulated delta range in meters.

State estimate naming conventions are as follows:
  * :code:`unix_time_millis` : (int) milliseconds that have elapsed
    since January 1, 1970 at midnight (midnight UTC) and not counting
    leapseconds.
  * :code:`x_rx_m` : (float) receiver ECEF x position in meters.
  * :code:`y_rx_m` : (float) receiver ECEF y position in meters.
  * :code:`z_rx_m` : (float) receiver ECEF z position in meters.
  * :code:`b_rx_m` : (float) receiver clock bias in meters.

GPS Time Conversions
--------------------

    * The GPS Week starts at 12:00am on Sunday morning
    * Converting GPS millis since gps start -> UTC
    * UTC to GPS week / time of the week
    * GPS week / time into UTC
    * added info about when GPS week starts/ends
    * GPS week rollover discussion
    * leap second discussion (when was it last changed?) From 18 to 19 on
      December 2016(??)
    * Other common errors?
    * :code:`time_of_ephemeris_millis` : (int) time of ephemeris as
      number of milliseconds since the start of the GPS epoch,
      January 6th, 1980.


Module Level Function References
--------------------------------
All functions and classes are fully documented in the linked
documentation below.

  .. toctree::
     :maxdepth: 2

     algorithms/modules
     parsers/modules
     utils/modules

Testing References
--------------------------------
All tests and test cases are fully documented in the linked
documentation below.

  .. toctree::
     :maxdepth: 2

     test_algorithms/modules
     test_parsers/modules
     test_utils/modules


Additional Indices
------------------

* :ref:`genindex`
* :ref:`modindex`
