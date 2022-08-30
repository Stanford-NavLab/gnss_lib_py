.. _reference:

Reference
=========

Package Architecture
--------------------

The gnss_lib_py package is broadly divided into the following sections.
Please choose the most appropriate location based on the descriptions
below when adding new features or functionality.

* :code:`algorithms` : This directory contains localization algorithms.
* :code:`parsers` : This directory contains functions to read and process various
  GNSS data/file types.
* :code:`utils` : This directory contains utilities used to handle
  GNSS measurements, time conversions, visualizations, satellite
  simulation, file operations, etc.

More information about currently available methods and the folder
organization can be found in the :ref:`organization subsection <organization>`.

Details about NavData Class
---------------------------

We use a custom class :code:`NavData` in :code:`gnss_lib_py` for storing
measurements and state estimates.
Along with the standard naming convention for measurements and
state estimates, the :code:`NavData` class provides modularity between
different datasets, algorithms and functions for visualization and metric
calculation.

Using our custom :code:`NavData` class has the following advantages:

* Our implementation uses string labels to intuitively access and set
  values in the underlying array. Eg. :code:`data['row_name'] = row_values`.
  This prevents mistakenly accessing the wrong row while using
  measurements
* Our implementation uses :code:`np.ndarray` as the underlying data
  data storage object, which is faster than :code:`pd.DataFrame`. The
  speed increase over Pandas' :code:`pd.DataFrame` is illustrated in the
  :code:`timing_comparisons` `example notebook <https://gnss-lib-py.readthedocs.io/en/latest/reference/timing_comparisons_notebook.html>`__.

  .. toctree::
     :maxdepth: 1
     :hidden:

     timing_comparisons_notebook
* We have implemented custom methods for adding new rows (measurement
  types), adding new columns (time stamps of data), deleting rows and
  columns and creating copies, including subsets
* We have also implemented custom methods for returning subsets where
  given equality and inequalities are satisfied
* :code:`NavData` also has implementations to loop over a larger
  :code:`NavData` column-wise and loop over subsets grouped over time
* :code:`NavData` also supports string valued entries, which are stored
  numerically in the underlying array. We provide methods so that
  accessing string valued rows takes strings as inputs and outputs.
  Users don't have to worry about the internal handling of string values

However, :code:`NavData` is maintained as part of :code:`gnss_lib_py`
and might not have all desired functionality that more mature libraries,
like pandas and numpy might have.
As a workaround, since the underlying storage is in :code:`np.ndarray`
and we provide functions for handling strings, you can implement your
own methods using a combination of numpy methods and :code:`NavData`
methods.


Standard Naming Conventions
---------------------------

In large part our conventions follow from the naming patterns in Google's
derived datasets for the `Google Decimeter challenge <https://www.kaggle.com/competitions/smartphone-decimeter-2022/data>`_



GNSS measurement naming conventions are as follows:

* :code:`trace_name` : (string) name for the trace
* :code:`rx_name` : (string) name for the receiver device
* :code:`gps_millis` : (float) milliseconds that have elapsed
  since the start of the GPS epoch on January 6th, 1980.
  :code:`gps_millis` is the common method for time that we expect
  in many functions and must be created to use some of the algorithms.
* :code:`gps_week` : (int) GPS weeks since the start of the GPS epoch
  on January 6th, 1980. The `NOAA CORS website <https://geodesy.noaa.gov/CORS/Gpscal.shtml>`__
  maintains a helpful reference calendar.
* :code:`gps_tow` : (float) time of receiving signal as measured by
  the receiver in seconds since start of GPS week (Sunday at midnight).
  This time includes leap seconds
* :code:`unix_millis` : (int) milliseconds that have elapsed
  since January 1, 1970 at midnight (midnight UTC) and not counting
  leapseconds.
* :code:`gnss_id` : (string) GNSS identification using the constellation
  name in lowercase, possible options are :code:`gps`, :code:`galileo`
  :code:`glonass`, :code:`qzss`, :code:`sbas`, etc.
* :code:`sv_id` : (int) satellite vehicle identification number
* :code:`signal_type` (string) Identifier for signal type, eg.
  :code:`l1` for GPS L1 signal, :code:`e5` for Galileo's E5 signal or
  :code:`b1i` for BeiDou's B1I signal. The string is expected to
  consist of lowercase letters and numbers.
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
* :code:`raw_pr_sigma_m` : (float) uncertainty (standard deviation) of
  the raw, uncorrected pseuodrange in meters.
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

* :code:`gps_millis` : (float) milliseconds that have elapsed
  since the start of the GPS epoch on January 6th, 1980.
  :code:`gps_millis` is the common method for time that we expect
  in many functions and must be created to use some of the algorithms.
* :code:`x_rx_m` : (float) receiver ECEF x position estimate in meters.
* :code:`y_rx_m` : (float) receiver ECEF y position estimate in meters.
* :code:`z_rx_m` : (float) receiver ECEF z position estimate in meters.
* :code:`b_rx_m` : (float) receiver clock bias in meters.
* :code:`lat_rx_deg` : (float) receiver latitude position estimate in
  degrees.
* :code:`lon_rx_deg` : (float) receiver longitude position estimate in
  degrees.
* :code:`alt_rx_m` : (float) receiver altitude position estimate in
  meters. Referenced to the WGS-84 ellipsoid.


Ground truth naming conventions are as follows:

* :code:`gps_millis` : (float) milliseconds that have elapsed
  since the start of the GPS epoch on January 6th, 1980.
  :code:`gps_millis` is the common method for time that we expect
  in many functions and must be created to use some of the algorithms.
* :code:`x_gt_m` : (float) receiver ECEF x ground truth position in
  meters.
* :code:`y_gt_m` : (float) receiver ECEF y ground truth position in
  meters.
* :code:`z_gt_m` : (float) receiver ECEF z ground truth position in
  meters.
* :code:`lat_gt_deg` : (float) receiver ground truth latitude in
  degrees.
* :code:`lon_gt_deg` : (float) receiver ground truth longitude in
  degrees.
* :code:`alt_gt_m` : (float) receiver ground truth altitude in meters.
  Referenced to the WGS-84 ellipsoid.


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
