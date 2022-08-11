"""Functions to process RINEX2 observable .O files.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "09 July 2022"

from datetime import datetime, timedelta, timezone

import numpy as np
import georinex as gr

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.time_conversions import datetime_to_tow

class Rinex2Obs(NavData):
    """Class handling derived measurements from .O dataset.

    Inherits from Measurement().
    """
    def __init__(self, input_path):
        """RINEX-2 observable specific loading and preprocessing for NavData()

        Parameters
        ----------
        input_path : string
            Path to measurement csv file

        Returns
        -------
        pd_df : pd.DataFrame
            Loaded measurements with consistent column names

        Notes
        -----
        TOCLARIFY:
        (1) Not sure if it is okay to add operations in init before calling postprocess
        Or perhaps should I change it to take input as obs_gnss.attrs and update x_gt_m,
        y_gt_m and z_gt_m
        (2) Currently the computed dgnss corrections are only overwriting derived_data
        ["corr_pr_m"] and not getting stored anywhere in rinex2 object. should I?
        If so, by which name? If I store the advantage is that compute will be faster,
        no need to re-compute if the time-step matches.
        (3) Can any of the fields in NavData be string array? couldn't get that to work
        (4) For rxdatetime, prefer directly getting it from this class instead of recom
        puting. is there a way to do it?
        (5) Need to find a way to stop the pandas_to_dataframe from truncating time
        values
        (6) Not sure if there is a way to truncate the basestation data .o file, is
        this size too long?
        (7) pylint shows "too few public methods". Any advice on how I can fix?
        (8) How to extract only few time instants in NavData class?

        References
        ----------
        .. [1]  https://geodesy.noaa.gov/UFCORS/ Accessed as of August 2, 2022

        """
        obs_gnss = gr.load(input_path)
        obs_gnss_df = obs_gnss.to_dataframe()
        obs_gnss_df_single = obs_gnss_df.reset_index()

        col_map = self._column_map()
        obs_gnss_df_single.rename(columns=col_map, inplace=True)
        super().__init__(pandas_df=obs_gnss_df_single)

        self['x_gt_m'] = obs_gnss.attrs['position'][0] * np.ones(np.shape(self['raw_pr_m']))
        self['y_gt_m'] = obs_gnss.attrs['position'][1] * np.ones(np.shape(self['raw_pr_m']))
        self['z_gt_m'] = obs_gnss.attrs['position'][2] * np.ones(np.shape(self['raw_pr_m']))

#         times = pd.to_datetime(obs_gnss_df_single.time).values

        self.postprocess()

    def postprocess(self):
        """RINEX 2 observable specific postprocessing for NavData()

        Notes
        -----
        TOCLARIFY:
        (1) Need to update this time function with something already in time conversions or
        perhaps modify timeconversions to create a rinex suitable one, i.e., nanoseconds and
        start epoch from 1970 jan 1 instead

        """
        times = [datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=tx*1e-9) \
                         for tx in self['time'][0]]
        extract_gps_time = np.transpose([list(datetime_to_tow(tt, \
                                              convert_gps=False)) for tt in times])
        self['gps_week'] = extract_gps_time[0]
        self['gps_tow'] = extract_gps_time[1]

        gnss_id_map = {'E': 6, 'G': 1, 'R': 3}

        self['sv_id'] = np.empty(len(self['sv'][0]))
        self['gnss_id'] = np.empty(len(self['sv'][0]))

        for i in range(len(self['sv'][0])):
            self['sv_id',i] = int(self['sv'][0][i][1:3])
            self['gnss_id',i] = gnss_id_map[self['sv'][0][i][0]]

    @staticmethod
    def _column_map():
        """Map of column names from loaded to gnss_lib_py standard

        Returns
        -------
        col_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        col_map = {'C1' : 'raw_pr_m',
                   'D1' : 'raw_dp_hz',
                   'S1' : 'cn0_dbhz'
                   }
        return col_map
