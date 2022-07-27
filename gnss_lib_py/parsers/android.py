"""Functions to process Android measurements.

"""

__authors__ = "Shubh Gupta, Adam Dai, Ashwin Kanhere"
__date__ = "02 Nov 2021"

import os
import csv

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from gnss_lib_py.parsers.measurement import Measurement
from gnss_lib_py.core.ephemeris import datetime2tow
from gnss_lib_py.core.coordinates import geodetic2ecef


class AndroidDerived(Measurement):
    """Class handling derived measurements from Android dataset.

    Inherits from Measurement().
    """
    def __init__(self, input_path):
        """Android specific loading and preprocessing for Measurement()

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

        super().__init__(pandas_df=pd_df)
        self.postprocess()

    def postprocess(self):
        """Android derived specific postprocessing for Measurement()

        Notes
        -----
        Adds corrected pseudoranges to measurements. Corrections
        implemented from https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data
        retrieved on 12 January, 2022
        """
        pr_corrected = self['raw_pr_m', :] \
                     + self['b_sv_m', :] \
                     - self['intersignal_bias_m', :] \
                     - self['tropo_delay_m', :] \
                     - self['iono_delay_m', :]
        self['corr_pr_m'] = pr_corrected

        times = [datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=tx*1e-3) \
                         for tx in self['millisSinceGpsEpoch',:][0]]
        extract_gps_time = np.transpose([list(datetime2tow(tt)) for tt in times])
        self['gps_week'] = extract_gps_time[0]
        self['gps_tow'] = extract_gps_time[1]

        sat_times = [datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=tx*1e-9) \
                         for tx in self['receivedSvTimeInGpsNanos',:][0]]
        sat_extract_gps_time = np.transpose([list(datetime2tow(tt)) for tt in sat_times])
        print(sat_extract_gps_time[0])
        if np.array_equal(extract_gps_time[0], sat_extract_gps_time[0]):
            self['tx_sv_tow'] = sat_extract_gps_time[1]
        else: 
            print('GPS weeks do not match!')
        
    @staticmethod
    def _column_map():
        """Map of column names from loaded to gnss_lib_py standard

        Returns
        -------
        col_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        col_map = {'collectionName' : 'trace_name',
                   'phoneName' : 'rx_name',
                   'constellationType' : 'gnss_id',
                   'svid' : 'sv_id',
                   'signalType' : 'signal_type',
                   'xSatPosM' : 'x_sv_m',
                   'ySatPosM' : 'y_sv_m',
                   'zSatPosM' : 'z_sv_m',
                   'xSatVelMps' : 'vx_sv_mps',
                   'ySatVelMps' : 'vy_sv_mps',
                   'zSatVelMps' : 'vz_sv_mps',
                   'satClkBiasM' : 'b_sv_m',
                   'satClkDriftMps' : 'b_dot_sv_mps',
                   'rawPrM' : 'raw_pr_m',
                   'rawPrUncM' : 'raw_pr_sigma_m',
                   'isrbM' : 'intersignal_bias_m',
                   'ionoDelayM' : 'iono_delay_m',
                   'tropoDelayM' : 'tropo_delay_m',
                   }
        return col_map

class AndroidGroundTruth(Measurement):
    """Class handling derived measurements from Android dataset.

    Inherits from Measurement().
    """
    def __init__(self, input_path):
        """Android specific loading and preprocessing for Measurement()

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
        super().__init__(pandas_df=pd_df)

        self.postprocess()

    def postprocess(self):
        """Android derived specific postprocessing for Measurement()

        Notes
        -----
        Adds corrected pseudoranges to measurements. Corrections
        implemented from https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data
        retrieved on 12 January, 2022
        """
        times = [datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=tx*1e-3) \
                         for tx in self['millisSinceGpsEpoch',:][0]]
        extract_gps_time = np.transpose([list(datetime2tow(tt)) for tt in times])
        self['gps_week'] = extract_gps_time[0]
        self['gps_tow'] = extract_gps_time[1]

        gt_LLA = np.transpose(np.vstack([self['latDeg'], self['lngDeg'], self['heightAboveWgs84EllipsoidM']]))        
        gt_ECEF = geodetic2ecef(gt_LLA)
        self["x_gt_m"] = gt_ECEF[:,0]
        self["y_gt_m"] = gt_ECEF[:,1]
        self["z_gt_m"] = gt_ECEF[:,2]
        

class AndroidRawImu(Measurement):
    """Class handling IMU measurements from raw Android dataset.

    Inherits from Measurement().
    """
    def __init__(self, input_path, group_time=10):
        self.group_time = group_time
        pd_df = self.preprocess(input_path)
        super().__init__(pandas_df=pd_df)


    def preprocess(self, input_path):
        """Read Android raw file and produce IMU dataframe objects

        Parameters
        ----------
        input_path : string
            File location of data file to read.

        Returns
        -------
        accel : pd.DataFrame
            Dataframe that contains the accel measurements from the log.
        gyro : pd.DataFrame
            Dataframe that contains the gyro measurements from the log.

        """
        with open(input_path, encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0][0] == '#':
                    if 'Accel' in row[0]:
                        accel = [row[1:]]
                    elif 'Gyro' in row[0]:
                        gyro = [row[1:]]
                else:
                    if row[0] == 'Accel':
                        accel.append(row[1:])
                    elif row[0] == 'Gyro':
                        gyro.append(row[1:])

        accel = pd.DataFrame(accel[1:], columns = accel[0], dtype=np.float64)
        gyro = pd.DataFrame(gyro[1:], columns = gyro[0], dtype=np.float64)

        #Drop common columns from gyro and keep values from accel
        gyro.drop(columns=['utcTimeMillis', 'elapsedRealtimeNanos'], inplace=True)
        measurements = pd.concat([accel, gyro], axis=1)
        #NOTE: Assuming pandas index corresponds to measurements order
        #NOTE: Override times of gyro measurements with corresponding
        # accel times
        measurements.rename(columns=self._column_map(), inplace=True)
        return measurements

    @staticmethod
    def _column_map():
        col_map = {'AccelXMps2' : 'acc_x_mps2',
                   'AccelYMps2' : 'acc_y_mps2',
                   'AccelZMps2' : 'acc_z_mps2',
                   'GyroXRadPerSec' : 'ang_vel_x_radps',
                   'GyroYRadPerSec' : 'ang_vel_y_radps',
                   'GyroZRadPerSec' : 'ang_vel_z_radps',
                   }
        return col_map


class AndroidRawFixes(Measurement):
    """Class handling location fix measurements from raw Android dataset.

    Inherits from Measurement().
    """
    def __init__(self, input_path):
        pd_df = self.preprocess(input_path)
        super().__init__(pandas_df=pd_df)


    def preprocess(self, input_path):
        """Read Android raw file and produce location fix dataframe objects

        Parameters
        ----------
        input_path : string
            File location of data file to read.

        Returns
        -------
        fix_df : pd.DataFrame
            Dataframe that contains the location fixes from the log.

        """
        with open(input_path, encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0][0] == '#':
                    if 'Fix' in row[0]:
                        android_fixes = [row[1:]]
                else:
                    if row[0] == 'Fix':
                        android_fixes.append(row[1:])

        fix_df = pd.DataFrame(android_fixes[1:], columns = android_fixes[0])
        return fix_df


def make_csv(input_path, output_directory, field, show_path=False):
    """Write specific data types from a GNSS android log to a CSV.

    Parameters
    ----------
    input_path : string
        File location of data file to read.
    output_directory : string
        Directory where new csv file should be created
    fields : list of strings
        Type of data to extract. Valid options are either "Raw",
        "Accel", "Gyro", "Mag", or "Fix".

    Returns
    -------
    output_path : string
        New file location of the exported CSV.

    Notes
    -----
    Based off of MATLAB code from Google's gps-measurement-tools
    repository: https://github.com/google/gps-measurement-tools. Compare
    with MakeCsv() in opensource/ReadGnssLogger.m

    """
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, field + ".csv")
    with open(output_path, 'w', encoding="utf8") as out_csv:
        writer = csv.writer(out_csv)
        with open(input_path, 'r', encoding="utf8") as in_txt:
            for line in in_txt:
                # Comments in the log file
                if line[0] == '#':
                    # Remove initial '#', spaces, trailing newline
                    # and split using commas as delimiter
                    line_data = line[2:].rstrip('\n').replace(" ","").split(",")
                    if line_data[0] == field:
                        writer.writerow(line_data[1:])
                # Data in file
                else:
                    # Remove spaces, trailing newline and split using commas as delimiter
                    line_data = line.rstrip('\n').replace(" ","").split(",")
                    if line_data[0] == field:
                        writer.writerow(line_data[1:])
    if show_path:
        print(output_path)

    return output_path
