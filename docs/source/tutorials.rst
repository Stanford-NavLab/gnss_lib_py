.. _tutorials:

Tutorials
=========


How to Load Android Derived Dataset
-----------------------------------

This example shows how to quickly load in a Android derived dataset
file into the :code:`gnss_lib_py` framework.

.. code-block:: python

    from gnss_lib_py.parsers.android import AndroidDerived
    derived = AndroidDerived(derived_dataset_path)

How to Create a New Measurement Child Class
-------------------------------------------
The modular and versatile functionality of this :code:`gnss_lib_py`
repository is enabled by loading all measurement data types into a
standard Python `Measurement class <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/gnss_lib_py/parsers/measurement.py>`__.
If your measurements use a file type not already supported in the list
on our :ref:`main page<mainpage>`, then you will need to create a new
child measurement Python class. This tutorial will guide you on how to
set up your custom Python class. Once complete, please feel free to
submit a pull request to our GitHub repository so other users can also
make use of the added functionality.

1. Create preprocess.

2. Create postprocess

3. Create a new :code:`_column_map()` function that translates the
column names from the new measurement type into our standard names.
Those standard names are as follows:

    * :code:`collection_name`
    * :code:`receiver_device_name`
    * :code:`time_of_ephemeris_millis` : (int) time of ephemeris as
      number of milliseconds since the start of the GPS epoch,
      January 6th, 1980.
    * :code:`constellation_type` : (int) GNSS constellation type using
      the following mapping

        *  0 : UNKNOWN
        *  1 : GPS
        *  2 : SBAS
        *  3 : GLONASS
        *  4 : QZSS
        *  5 : BEIDOU
        *  6 : GALILEO
        *  7 : IRNSS

    * :code:`svid` : (int) satellite vehicle identification number
    * :code:`signal_type`
    * :code:`received_sv_time_nanos`
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
    * :code:`raw_pseudorange_m`
    * :code:`raw_pseudorange_uncertainty_m`
    * :code:`intersignal_bias_m`
    * :code:`iono_delay_m`
    * :code:`tropo_delay_m`

Your finished class might look something like:

.. code-block:: python

    from gnss_lib_py.parsers.measurement import Measurement

    class NewMeasurementType(Measurement):
    """Class handling derived measurements from Android dataset.
    Inherits from Measurement().
    """
    #NOTE: Inherits __init__() and isn't defined explicitly here because
    # no additional implementations/attributes are needed

    def preprocess(self, input_path):
        """Loading and preprocessing.

        Parameters
        ----------
        input_path : string
            Path to measurement csv file
        Returns
        -------
        pd_df : pd.DataFrame
            Loaded measurements with consistent column names
        """
        pd_df = pd.read_csv(input_path)
        col_map = self._column_map()
        pd_df.rename(columns=col_map, inplace=True)
        return pd_df

    def postprocess(self):
        """Postprocessing.

        """
        pass

    @staticmethod
    def _column_map():
        """Map of column names from loaded to gnss_lib_py standard

        Returns
        -------
        col_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        col_map = {'millisSinceGpsEpoch' : 'toeMillis',
                'svid' : 'PRN',
                'xSatPosM' : 'x_sat_m',
                'ySatPosM' : 'y_sat_m',
                'zSatPosM' : 'z_sat_m',
                'xSatVelMps' : 'vx_sat_mps',
                'ySatVelMps' : 'vy_sat_mps',
                'zSatVelMps' : 'vz_sat_mps',
                'satClkBiasM' : 'b_sat_m',
                'satClkDriftMps' : 'b_dot_sat_mps',
                }
        return col_map



Visualize your Data
-------------------

Examples of how to visualize data.
