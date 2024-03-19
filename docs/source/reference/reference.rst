.. _reference:

Reference
=========

General GNSS References
-----------------------

The GNSS in ``gnss_lib_py`` stands for Global Navigation Satellite
System(s). The following is a list of references that might be helpful
in learning more about GNSS:

* *Global Positioning System: Signals, Measurements, and Performance* by
  Pratap Misra and Per Enge.
* *Positioning, Navigation, and Timing Technologies in the 21st Century*
  edited by Y. Jade Morton, Frank van Diggelen, James J. Spilker Jr,
  Bradford W. Parkinson, Sherman Lo, and Grace Gao.
* *Understanding GPS/GNSS: Principles and Applications* edited by
  Elliott D. Kaplan and Christopher J. Hegarty.
* "GPS: An Introduction to Satellite Navigation" by Frank van Diggelen
  found on `YouTube <https://www.youtube.com/playlist?list=PLGvhNIiu1ubyEOJga50LJMzVXtbUq6CPo>`__:
  a playlist of online classes that teach a broad overview of GNSS
  topics.
*  "GPS" by Bartosz Ciechanowski found at
   `ciechanow.ski/gps <https://ciechanow.ski/gps/>`__:
   a visually-appealing and interactive blog post about some of the
   basic principles of GNSS positioning.

Reference Documents for GNSS Standards
--------------------------------------

GNSS constellations and receivers use standardized file formats to transfer
information such as estimated receiver coordinates, broadcast ephemeris
parameters, and precise ephimerides.
The parsers in ``gnss_lib_py`` are based on standard documentation for
the GNSS constellations and file types, which are listed below along with
their use in ``gnss_lib_py``.

  * *Rinex v2.11* (`version format document <https://geodesy.noaa.gov/corsdata/RINEX211.txt>`__
    retrieved on 2nd July, 2023): for parsing broadcast navigation ephimerides.
  * *Rinex v3.05* (`version format document <https://files.igs.org/pub/data/format/rinex305.pdf>`__
    retrieved on 2nd July, 2023): for parsing broadcast navigation ephimerides.
  * *Rinex v4.00* (`version format document <https://files.igs.org/pub/data/format/rinex_4.00.pdf>`__
    retrieved on 2nd July, 2023): currently not supported by ``gnss_lib_py``.
  * *NMEA* (`reference manual <https://www.sparkfun.com/datasheets/GPS/NMEA%20Reference%20Manual-Rev2.1-Dec07.pdf>`__
    retrieved on 23rd June, 2023): for parsing NMEA files with GGA and RMC messages.
  * *SP3*: used to determine SV positions for precise
  * *GLONASS ICD* (retrieved from this `link <https://www.unavco.org/help/glossary/docs/ICD_GLONASS_4.0_(1998)_en.pdf>`__
    retrieved on 27th June, 2023): for determining GLOASS SV states from
    broadcast satellite positions, velocities, and accelerations.


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


Timing Conventions
------------------

We use four different time formats in :code:`gnss_lib_py`:

* :code:`gps_millis` : The number of milliseconds that have elapsed since
  the start of the GPS epoch on January 6th, 1980. This time format is
  continuous and is not adjusted with leap seconds. This time is stored
  as a single number that is a :code:`double int` or :code:`float`,
  depending on the context.
* :code:`unix_millis` : The number of milliseconds that have elapsed since
  the start of the Unix epoch on January 1st, 1970. This time format is
  not continuous and is adjusted with leap seconds. This time is also
  stored as a single number that is a :code:`double int` or :code:`float`.
* :code:`utc_timestamp` : UTC time, which is stored as a timestamp using
  the :code:`datetime` library. This time format is not continuous and is
  adjusted with leap seconds.
* :code:`gps_week` and :code:`gps_tow` : The GPS week since the start of
  the GPS epoch on January 6th, 1980 and the time of that week in seconds.

Of these four time formats, we use :code:`gps_millis` as the default
time that measurements and state estimates correspond to. Conversions
between all these time formats are provided in the
:code:`utils/time_conversion.py` file.

Between these four time formats, all major applications of GNSS-based
state estimation should be covered and any of these time formats can be
used interchangeably.


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
  :code:`glonass`, :code:`beidou`, :code:`qzss`, :code:`sbas`,
  :code:`irnss`, etc.
* :code:`sv_id` : (int) satellite vehicle identification number
* :code:`gnss_sv_id` : (string) combination of :code:`gnss_id` and :code:`sv_id`
  in a three character string. The first character is the upper case
  letter for the satellite system identifier defined in the RINEX 3.04
  specification (e.g. G for gps, R for glonass, E for galileo,
  C for Beidou, etc.) followed by a two digit SV ID.
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
* :code:`el_sv_deg` : (float) Elevation of satellite in degrees in
  relation to the receiver's position.
* :code:`az_sv_deg` : (float) Azimuth of satellite in degrees in
  relation to the receiver's position.
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
* :code:`v_rx_mps` : (float) receiver total velocity estimate in
  meters per second.
* :code:`vx_rx_mps` : (float) receiver ECEF x velocity estimate in
  meters per second.
* :code:`vy_rx_mps` : (float) receiver ECEF y velocity estimate in
  meters per second.
* :code:`vz_rx_mps` : (float) receiver ECEF z velocity estimate in
  meters per second.
* :code:`ax_rx_mps2` : (float) receiver ECEF x acceleration estimate in
  meters per second squared.
* :code:`a_rx_mps2` : (float) receiver total acceleration estimate in
  meters per second squared.
* :code:`ay_rx_mps2` : (float) receiver ECEF y acceleration estimate in
  meters per second squared.
* :code:`az_rx_mps2` : (float) receiver ECEF z acceleration estimate in
  meters per second squared.
* :code:`b_rx_m` : (float) receiver clock bias in meters.
* :code:`b_dot_rx_mps` : (float) receiver clock bias drift rate in meters
  per second.
* :code:`lat_rx_deg` : (float) receiver latitude position estimate in
  degrees.
* :code:`lon_rx_deg` : (float) receiver longitude position estimate in
  degrees.
* :code:`alt_rx_m` : (float) receiver altitude position estimate in
  meters. Referenced to the WGS-84 ellipsoid.
* :code:`heading_rx_rad` : (float) receiver heading estimate in radians,
  clockwise from North, where to 0 radians is North, pi/2
  radians is East and so on.
  Assumed to be radians in the range between 0 and 2pi.

Receiver ground truth naming conventions are as follows:

* :code:`gps_millis` : (float) milliseconds that have elapsed
  since the start of the GPS epoch on January 6th, 1980.
  :code:`gps_millis` is the common method for time that we expect
  in many functions and must be created to use some of the algorithms.
* :code:`x_rx_gt_m` : (float) receiver ECEF x ground truth position in
  meters.
* :code:`y_rx_gt_m` : (float) receiver ECEF y ground truth position in
  meters.
* :code:`z_rx_gt_m` : (float) receiver ECEF z ground truth position in
  meters.
* :code:`v_rx_gt_mps` : (float) receiver total velocity ground truth in
  meters per second.
* :code:`vx_rx_gt_mps` : (float) receiver ECEF x velocity ground truth
  in meters per second.
* :code:`vy_rx_gt_mps` : (float) receiver ECEF y velocity ground truth
  in meters per second.
* :code:`vz_rx_gt_mps` : (float) receiver ECEF z velocity ground truth
  in meters per second.
* :code:`a_rx_gt_mps2` : (float) receiver total acceleration estimate in
  meters per second squared.
* :code:`ax_rx_gt_mps2` : (float) receiver ECEF x acceleration ground truth
  in meters per second squared.
* :code:`ay_rx_gt_mps2` : (float) receiver ECEF y acceleration ground truth
  in meters per second squared.
* :code:`az_rx_gt_mps2` : (float) receiver ECEF z acceleration ground truth
  in meters per second squared.
* :code:`lat_rx_gt_deg` : (float) receiver ground truth latitude in
  degrees.
* :code:`lon_rx_gt_deg` : (float) receiver ground truth longitude in
  degrees.
* :code:`alt_rx_gt_m` : (float) receiver ground truth altitude in meters.
  Referenced to the WGS-84 ellipsoid.
* :code:`heading_rx_gt_rad` : (float) receiver heading ground truth in
  radians, clockwise from North, where to 0 radians is North, pi/2
  radians is East and so on.
  Assumed to be radians in the range between 0 and 2pi.

Module Level Function References
--------------------------------
All functions and classes are fully documented in the linked
documentation below.

  .. toctree::
     :maxdepth: 2

     algorithms/modules
     navdata/modules
     parsers/modules
     utils/modules
     visualizations/modules

Testing References
--------------------------------
All tests and test cases are fully documented in the linked
documentation below.

  .. toctree::
     :maxdepth: 2

     test_algorithms/modules
     test_navdata/modules
     test_parsers/modules
     test_utils/modules
     test_visualizations/modules


Additional Indices
------------------

* :ref:`genindex`
* :ref:`modindex`
