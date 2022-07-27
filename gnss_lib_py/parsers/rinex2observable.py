"""Functions to process RINEX2 observable .O files.

"""

__authors__ = "Sriramya Bhamidipati"
__date__ = "09 July 2022"

import os
import csv

import numpy as np
import pandas as pd
import georinex as gr
from datetime import datetime, timedelta, timezone

from gnss_lib_py.parsers.measurement import Measurement
from gnss_lib_py.core.ephemeris import datetime2tow


class rinex2(Measurement): #Rinex-> capital
    """Class handling derived measurements from .O dataset.

    Inherits from Measurement().
    """
    def __init__(self, input_path):
        """RINEX 2 observable specific loading and preprocessing for Measurement()

        Parameters
        ----------
        input_path : string
            Path to measurement csv file

        Returns
        -------
        pd_df : pd.DataFrame
            Loaded measurements with consistent column names
        """
        obs_gnss = gr.load(input_path) 
        obs_gnss_df = obs_gnss.to_dataframe()
        obs_gnss_df_single = obs_gnss_df.reset_index()
        
        col_map = self._column_map()
        obs_gnss_df_single.rename(columns=col_map, inplace=True)
        super().__init__(pandas_df=obs_gnss_df_single)
#         print(obs_gnss_df_single)
#         print(obs_gnss_df_single['time'])
        
        print(obs_gnss.attrs['position'])

        self['x_gt_m'] = obs_gnss.attrs['position'][0] * np.ones(np.shape(self['raw_pr_m']))
        self['y_gt_m'] = obs_gnss.attrs['position'][1] * np.ones(np.shape(self['raw_pr_m']))
        self['z_gt_m'] = obs_gnss.attrs['position'][2] * np.ones(np.shape(self['raw_pr_m']))
        self["delta_pr_m"] = np.nan * np.ones(np.shape(self['raw_pr_m']))

        self.postprocess()

        # why does it have to be stack length same as array!
        # from labels such as x_rx_m and x_sv_m it is not clear which coordinate frames x, y, z are and what to do for creating fields that are in NED or LLA frames
        # a very high-level comment: I was relateing "Measurement" to items like pseudoranges, carrier phase. It took me a lot time to understand that measurement is a generic class that can be used to create instances of ephemeris, ground truth etc. Maybe name can be something other than "measurement" to avoid confusion
        # TODO: rinex2observable does not account for rinex files with moving data, as I am not sure what obs_gnss.attrs['position'] would like for moving files
        # solve_wls does not have functionality to choose specific or subset of constellation (GPS, GLONASS), frequency (L1, L5) and measurement (raw, dgnss, corr, carrier smoothed pseudorange)
        # corr_pr_m in solve_wls and wls are misleading as it is not clear what to do for dgnss, carrier smoothed pseudoranges; should I rewrite the corr_pr_m or create additional fields like delta_pr_m or carrsmth_pr_m etc. 
        # It seems more intuitive to base solve_wls on gps_time rather than millisSinceGpsEpoch, if you want solve_wls to have more generic touch and not android derived specific
        # As a follow-up not clear if solve_wls is android specific or it is supposed to be generic function that can be used for any dataset
        # A high-level remark is it would be better for solve_wls to have LLA output of estimated state. I wanted to plot the output on a google map to see how the trajectory looks like, ECEF is not intuitive to interpret.
        # Right now, I create a separate dgnsscorrections file. How does it need to be placed in file structure, separate file or part of snapshot.py, residuals.py?
        # TODO: To incorporate corrections for other constellations, need to include sp3 files. Right now considering .n files as input. 
        # To compute dgps corrections, I am reading in ephemeris based gps satellite information, since android derived only gives for specific time instants. What do you guys think about my design choice?
        # inconsistency in single/double quotations for fields: for example: android has self['corr_pr_m'] while snapshot has self["corr_pr_m"].
        # Right now, based stations have 30s data frequency. How do incorporate this data, right now I am just estimating nearest timestamp and using that value. Other options are interpolating to specific time instant, picking nearest 2-3 and averaging, etc. any thoughts?
        # how are you checking for if solve_wls has nan for pseudoranges?
        # I wanted to create a self['gps_datetime'] array float() argument must be a string or a number, not 'datetime.datetime'
        # TODO: With Adyasha's help, I found references to calculate carrier smoothed pseudoranges from ADR values. This can be the next functionality to be added to the repo.
        # TODO: The supplementary files folder of android datasets on kaggle has rinex .o files that can be parsed using georinex to extract carrier phase measurements. This can be used to develop carrier phase positioning.  Another potential functionality I am thinking of adding to the repo in near future. what do you guys think? 
        # For TRI project, used pyubx2 toolbox to parse .ubx files or real-time stream of data from ublox receiver. Another potential functionality I am thinking of adding to the repo in near future 
        # created new dgnss-wls branch so its easier for me to integrate. I can delete the dgnss branch in the next week or so.
        # used x_gt_m for ground truth is that okay? or a different name
        # used rinex2 and rinex2observable names. is that okay or what different name?
        # there seems to be an error. signal_type says need number? not string? 
        # what about the ones for which dgnss corrections not available : like different signal type
        # to extract android ground truth -> a new measurement class or add rows to derived_data?
        # currently rinex2obs has additional fields -> L3,C3,P3 etc. remove or keep?
        # look at the contributing guide for syntax
        
    def postprocess(self):
        """RINEX 2 observable specific postprocessing for Measurement()

        Notes
        -----
        
        """
        times = [datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=tx*1e-9) \
                         for tx in self['time'][0]]
        extract_gps_time = np.transpose([list(datetime2tow(tt)) for tt in times])
        self['gps_week'] = extract_gps_time[0]
        self['gps_tow'] = extract_gps_time[1]

        gnss_id_map = {'E': 6, 'G': 1, 'R': 3}
        signal_type_map = {'E': 'GAL_E1', 'G': 'GPS_L1', 'R': 'GLO_G1'}

        self['sv_id'] = np.empty(len(self['sv'][0]))
        self['gnss_id'] = np.empty(len(self['sv'][0]))
        self['gnss_id'] = np.empty(len(self['sv'][0]))
        
        for i in range(len(self['sv'][0])): 
            self['sv_id',i] = int(self['sv'][0][i][1:3])
            self['gnss_id',i] = gnss_id_map[self['sv'][0][i][0]]
        
        # to be added later: x_gt_m, y_gt_m, z_gt_m, 
        # to be added later: x_sv_m, y_sv_m, z_sv_m
        # to be added later: delta_pr_m
        
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
    
