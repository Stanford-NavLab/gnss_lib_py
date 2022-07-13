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
        self.postprocess()

        # why does it have to be stack length same as array!
        # from labels not clear which coordinate x, y, z are and what to do for NED/LLA frames
        # name of measurement class can be something more generic
        # does not account for rinex files with moving data!
        # functionality to choose specific constellations?
        # corr_pr_m in solve_wls is misleading
        # better to use GPS_time in solve_wls instead
        # better for solve_wls to have LLA output as well.
        # should calculating dgps corrections be a part of wls_residuals, residuals, separate?
        # how to deal with gps satellite information -> eventually integrate with sp3, anything for timebeing?
        # inconsistency in single/double quotations for fields.
        # do not have faster corrections rate becoz of old data. any thoughts?
        # how are you checking for if solve_wls has nan for pseudoranges?
        # float() argument must be a string or a number, not 'datetime.datetime'
        # what kind of interpolation/format do you prefer for dGPS corrections
        # add carrier smoothed pseudoranges from ADR values.
        # there are .o files available for carrier phase positioning. -> what do you guys think? 
        # .ubx files -> 
        # wls-> should have different naming, not corr_pr_m. what about solve_wls too.
        # created new dgnss-wls branch so its easier for me to integrate.
        # used x_gt_m for ground truth is that okay?
        # signal_type says need number? not string? 
        # what about the ones for which dgnss corrections not available : like different signal type
        # android ground truth -> same class or different one?
        # currently rinex2obs has additional fields -> L3,C3,P3 etc. remove or keep?
        # supplemental files have .O 
        # corr_pr_m: based on dgnss, carrier smoothed pseudoranges, 
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
    
