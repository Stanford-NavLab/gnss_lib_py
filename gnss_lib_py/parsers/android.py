"""Functions to process Android measurements.

"""

__authors__ = "Ashwin Kanhere, Shubh Gupta, Adam Dai"
__date__ = "02 Nov 2021"


import os
import csv
import warnings

import numpy as np
import pandas as pd

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.utils.coordinates import wrap_0_to_2pi
from gnss_lib_py.utils.coordinates import geodetic_to_ecef
from gnss_lib_py.utils.coordinates import ecef_to_geodetic
from gnss_lib_py.utils.time_conversions import unix_to_gps_millis
from gnss_lib_py.utils.time_conversions import gps_to_unix_millis

class AndroidDerived2021(NavData):
    """Class handling derived measurements from Android dataset.

    Inherits from NavData().
    """
    def __init__(self, input_path, remove_timing_outliers=True):
        """Android specific loading and preprocessing

        Parameters
        ----------
        input_path : string
            Path to measurement csv file
        remove_timing_outliers : bool
            Flag for whether to remove measures that are too close or
            too far away in time. Code from the competition hosts used
            to implement changes. See note.

        Notes
        -----
        Removes duplicate rows using correction 5 from competition hosts
        implemented from https://www.kaggle.com/code/gymf123/tips-notes-from-the-competition-hosts/notebook
        retrieved on 10 August, 2022

        """
        pd_df = pd.read_csv(input_path)
        # Correction 1: Mapping _derived timestamps to previous timestamp
        # for correspondance with ground truth and Raw data
        derived_timestamps = pd_df['millisSinceGpsEpoch'].unique()
        mapper = dict(zip(derived_timestamps[1:],derived_timestamps[:-1]))
        pd_df = pd_df[pd_df['millisSinceGpsEpoch'] != derived_timestamps[0]]
        pd_df.replace({"millisSinceGpsEpoch" : mapper},inplace=True)

        # Correction 5 implemented verbatim from competition tips
        if remove_timing_outliers:
            delta_millis = pd_df['millisSinceGpsEpoch'] - pd_df['receivedSvTimeInGpsNanos'] / 1e6
            where_good_signals = (delta_millis > 0) & (delta_millis < 300)
            pd_df = pd_df[where_good_signals].copy()
            if np.all(~where_good_signals):
                warnings.warn("All data removed due to timing outliers,"
                            + " try setting remove_timing_outliers to"
                            + " False", RuntimeWarning)

        super().__init__(pandas_df=pd_df)

    def postprocess(self):
        """Android derived specific postprocessing.

        Adds corrected pseudoranges to measurements. Time step
        corrections implemented from dataset webpage [1]_ retrieved on
        10 August, 2022.

        Correlates constellation type numbers with corresponding
        constellation names. Mapping also comes from competition
        website [1]_.

        References
        ----------
        .. [1] https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data

        """
        pr_corrected = self['raw_pr_m'] \
                     + self['b_sv_m'] \
                     - self['intersignal_bias_m'] \
                     - self['tropo_delay_m'] \
                     - self['iono_delay_m']
        self['corr_pr_m'] = pr_corrected
        # rename gnss_id column to constellation type
        constellation_map = {0.:"unknown",
                             1.:"gps",
                             2.:"sbas",
                             3.:"glonass",
                             4.:"qzss",
                             5.:"beidou",
                             6.:"galileo",
                             7.:"irnss",
                            }
        self.replace(constellation_map, rows="gnss_id", inplace=True)

        # rename signal_type column to conform to standard convention
        signal_map = {"GPS_L1" : "l1",
                      "GPS_L5" : "l5",
                      "GAL_E1" : "e1",
                      "GAL_E5A" : "e5a",
                      "GLO_G1" : "g1",
                      "QZS_J1" : "j1",
                      "QZS_J5" : "j5",
                      "BDS_B1I" : "b1i",
                      "BDS_B1C" : "b1c",
                      "BDS_B2A" : "b2a",
                     }
        self.replace(signal_map, rows="signal_type", inplace=True)

    @staticmethod
    def _row_map():
        """Map of row names from loaded to gnss_lib_py standard

        Returns
        -------
        row_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        row_map = {'collectionName' : 'trace_name',
                   'phoneName' : 'rx_name',
                   'millisSinceGpsEpoch' : 'gps_millis',
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
        return row_map


class AndroidDerived2022(NavData):
    """Class handling derived measurements from Android dataset.

    Inherits from NavData().
    The row nomenclature for the new derived dataset has changed.
    We reflect this changed nomenclature in the _row_map() method.
    """

    def __init__(self, input_path):
        """Android specific loading and preprocessing

        Parameters
        ----------
        input_path : string
            Path to measurement csv file
        """
        super().__init__(csv_path=input_path)

    def postprocess(self):
        """Android derived specific postprocessing

        Notes
        -----
        Adds corrected pseudoranges to measurements. Time step corrections
        implemented from https://www.kaggle.com/c/google-smartphone-decimeter-challenge/data
        retrieved on 10 August, 2022
        """
        pr_corrected = self['raw_pr_m'] \
                     + self['b_sv_m'] \
                     - self['intersignal_bias_m'] \
                     - self['tropo_delay_m'] \
                     - self['iono_delay_m']
        self['corr_pr_m'] = pr_corrected

        # rename gnss_id column to constellation type
        constellation_map = {0.:"unknown",
                             1.:"gps",
                             2.:"sbas",
                             3.:"glonass",
                             4.:"qzss",
                             5.:"beidou",
                             6.:"galileo",
                             7.:"irnss",
                            }
        self.replace(constellation_map, rows="gnss_id", inplace=True)

        # rename signal_type column to conform to standard convention
        signal_map = {"GPS_L1" : "l1",
                      "GPS_L5" : "l5",
                      "GAL_E1" : "e1",
                      "GAL_E5A" : "e5a",
                      "GLO_G1" : "g1",
                      "QZS_J1" : "j1",
                      "QZS_J5" : "j5",
                      "BDS_B1I" : "b1i",
                      "BDS_B1C" : "b1c",
                      "BDS_B2A" : "b2a",
                     }
        self.replace(signal_map, rows="signal_type", inplace=True)

        # add gps milliseconds
        self["gps_millis"] = unix_to_gps_millis(self["unix_millis"])

    @staticmethod
    def _row_map():
        """Map of row names from loaded to gnss_lib_py standard

        Returns
        -------
        row_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        row_map = {'utcTimeMillis' : 'unix_millis',
                   'ConstellationType' : 'gnss_id',
                   'Svid' : 'sv_id',
                   'SignalType' : 'signal_type',
                   'SvPositionXEcefMeters' : 'x_sv_m',
                   'SvPositionYEcefMeters' : 'y_sv_m',
                   'SvPositionZEcefMeters' : 'z_sv_m',
                   'SvElevationDegrees' : 'el_sv_deg',
                   'SvAzimuthDegrees' : 'az_sv_deg',
                   'SvVelocityXEcefMetersPerSecond' : 'vx_sv_mps',
                   'SvVelocityYEcefMetersPerSecond' : 'vy_sv_mps',
                   'SvVelocityZEcefMetersPerSecond' : 'vz_sv_mps',
                   'SvClockBiasMeters' : 'b_sv_m',
                   'SvClockDriftMetersPerSecond' : 'b_dot_sv_mps',
                   'RawPseudorangeMeters' : 'raw_pr_m',
                   'RawPseudorangeUncertaintyMeters' : 'raw_pr_sigma_m',
                   'IsrbMeters' : 'intersignal_bias_m',
                   'IonosphericDelayMeters' : 'iono_delay_m',
                   'TroposphericDelayMeters' : 'tropo_delay_m',
                   'Cn0DbHz': 'cn0_dbhz',
                   'AccumulatedDeltaRangeMeters' : 'accumulated_delta_range_m',
                   'AccumulatedDeltaRangeUncertaintyMeters': 'accumulated_delta_range_sigma_m',
                   'WlsPositionXEcefMeters' : 'x_rx_m',
                   'WlsPositionYEcefMeters' : 'y_rx_m',
                   'WlsPositionZEcefMeters' : 'z_rx_m',
                   }
        return row_map


class AndroidGroundTruth2021(NavData):
    """Class handling ground truth from Android dataset.

    Inherits from NavData().
    """
    def __init__(self, input_path):
        """Android specific loading and preprocessing for NavData()

        Parameters
        ----------
        input_path : string
            Path to measurement csv file
        """

        super().__init__(csv_path=input_path)

        self.postprocess()

    def postprocess(self):
        """Android derived specific postprocessing for NavData()

        Notes
        -----
        Corrections incorporated from Kaggle notes hosted here:
        https://www.kaggle.com/code/gymf123/tips-notes-from-the-competition-hosts
        """
        # Correcting reported altitude
        self['alt_rx_gt_m'] = self['alt_rx_gt_m'] - 61.
        gt_lla = np.transpose(np.vstack([self['lat_rx_gt_deg'],
                                         self['lon_rx_gt_deg'],
                                         self['alt_rx_gt_m']]))
        gt_ecef = geodetic_to_ecef(gt_lla)
        self["x_rx_gt_m"] = gt_ecef[:,0]
        self["y_rx_gt_m"] = gt_ecef[:,1]
        self["z_rx_gt_m"] = gt_ecef[:,2]

        # convert bearing degrees to heading in radians
        self["heading_rx_gt_rad"] = np.deg2rad(self["heading_rx_gt_rad"])
        self["heading_rx_gt_rad"] = wrap_0_to_2pi(self["heading_rx_gt_rad"])

    @staticmethod
    def _row_map():
        """Map of row names from loaded ground truth to gnss_lib_py standard

        Returns
        -------
        row_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        row_map = {'latDeg' : 'lat_rx_gt_deg',
                   'lngDeg' : 'lon_rx_gt_deg',
                   'heightAboveWgs84EllipsoidM' : 'alt_rx_gt_m',
                   'millisSinceGpsEpoch' : 'gps_millis',
                   'speedMps' : 'v_rx_gt_mps',
                   'courseDegree' : 'heading_rx_gt_rad',
                }
        return row_map


class AndroidGroundTruth2022(AndroidGroundTruth2021):
    """Class handling ground truth from Android dataset.

    Inherits from AndroidGroundTruth2021().
    """

    def postprocess(self):
        """Android derived specific postprocessing for NavData()

        Notes
        -----
        """
        if np.any(np.isnan(self['alt_rx_gt_m'])):
            warnings.warn("Some altitude values were missing, using 0m ", RuntimeWarning)
            self['alt_rx_gt_m'] = np.nan_to_num(self['alt_rx_gt_m'])
        gt_lla = np.transpose(np.vstack([self['lat_rx_gt_deg'],
                                         self['lon_rx_gt_deg'],
                                         self['alt_rx_gt_m']]))
        gt_ecef = geodetic_to_ecef(gt_lla)
        self["x_rx_gt_m"] = gt_ecef[:,0]
        self["y_rx_gt_m"] = gt_ecef[:,1]
        self["z_rx_gt_m"] = gt_ecef[:,2]

        # add gps milliseconds
        self["gps_millis"] = unix_to_gps_millis(self['unix_millis'])

        # convert bearing degrees to heading in radians
        self["heading_rx_gt_rad"] = np.deg2rad(self["heading_rx_gt_rad"])
        self["heading_rx_gt_rad"] = wrap_0_to_2pi(self["heading_rx_gt_rad"])

    @staticmethod
    def _row_map():
        """Map row names from loaded data to gnss_lib_py standard

        Returns
        -------
        row_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        row_map = {'LatitudeDegrees' : 'lat_rx_gt_deg',
                   'LongitudeDegrees' : 'lon_rx_gt_deg',
                   'AltitudeMeters' : 'alt_rx_gt_m',
                   'SpeedMps' : 'v_rx_gt_mps',
                   'BearingDegrees' : 'heading_rx_gt_rad',
                   'UnixTimeMillis' : 'unix_millis',

                }
        return row_map

class AndroidRawImu(NavData):
    """Class handling IMU measurements from raw Android dataset.

    Inherits from NavData().
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

        accel = pd.DataFrame(accel[1:], columns = accel[0],
                             dtype=np.float64)
        gyro = pd.DataFrame(gyro[1:], columns = gyro[0],
                            dtype=np.float64)

        #Drop common columns from gyro and keep values from accel
        gyro.drop(columns=['utcTimeMillis', 'elapsedRealtimeNanos'],
                  inplace=True)
        measurements = pd.concat([accel, gyro], axis=1)
        #NOTE: Assuming pandas index corresponds to measurements order
        #NOTE: Override times of gyro measurements with corresponding
        # accel times
        return measurements

    @staticmethod
    def _row_map():
        row_map = {'AccelXMps2' : 'acc_x_mps2',
                   'AccelYMps2' : 'acc_y_mps2',
                   'AccelZMps2' : 'acc_z_mps2',
                   'GyroXRadPerSec' : 'ang_vel_x_radps',
                   'GyroYRadPerSec' : 'ang_vel_y_radps',
                   'GyroZRadPerSec' : 'ang_vel_z_radps',
                   }
        return row_map


class AndroidRawFixes(NavData):
    """Class handling location fix measurements from raw Android dataset.

    Inherits from NavData().
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
    if not os.path.isdir(output_directory): #pragma: no cover
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
    if show_path: #pragma: no cover
        print(output_path)

    return output_path

def solve_kaggle_baseline(navdata):
    """Convert Decimeter challenge baseline into state_estimate.

    The baseline solution was provided in 2022, but not in 2021.

    Parameters
    ----------
    navdata : gnss_lib_py.parsers.android.AndroidDerived2022
        Instance of the AndroidDerived2022 class.

    Returns
    -------
    state_estimate : gnss_lib_py.parsers.navdata.NavData
        Baseline state estimate.

    """

    columns = ["unix_millis",
               "x_rx_m",
               "y_rx_m",
               "z_rx_m",
               ]
    navdata.in_rows(columns)
    data_df = (navdata.pandas_df().drop_duplicates(subset='unix_millis')[columns]
               .reset_index(drop=True))
    lat,lon,alt = np.transpose(ecef_to_geodetic(data_df[["x_rx_m",
                                                         "y_rx_m",
                                                         "z_rx_m",
                                                         ]].to_numpy()))

    state_estimate = NavData()
    state_estimate["gps_millis"] = unix_to_gps_millis(
                                     data_df["unix_millis"].to_numpy())
    state_estimate["lat_rx_deg"] = lat
    state_estimate["lon_rx_deg"] = lon
    state_estimate["alt_rx_deg"] = alt

    return state_estimate

def prepare_kaggle_submission(state_estimate, trip_id="trace/phone"):
    """Converts from gnss_lib_py receiver state to Kaggle submission.

    Parameters
    ----------
    state_estimate : gnss_lib_py.parsers.navdata.NavData
        Estimated receiver position in latitude and longitude as an
        instance of the NavData class with the following
        rows: ``gps_millis``, ``lat_rx*_deg``, ``lon_rx*_deg``.
    trip_id : string
        Value for the tripId column in kaggle submission which is a
        fusion of the data and phone type.

    Returns
    -------
    output : gnss_lib_py.parsers.navdata.NavData
        NavData structure ready for Kaggle submission.

    """

    state_estimate.in_rows("gps_millis")
    wildcards = state_estimate.find_wildcard_indexes(["lat_rx*_deg",
                            "lon_rx*_deg"],max_allow = 1)

    output = NavData()
    output["tripId"] = np.array([trip_id] * state_estimate.shape[1])
    output["UnixTimeMillis"] = gps_to_unix_millis(state_estimate["gps_millis"])
    output.orig_dtypes["UnixTimeMillis"] = np.int64
    output["LatitudeDegrees"] = state_estimate[wildcards["lat_rx*_deg"]]
    output["LongitudeDegrees"] = state_estimate[wildcards["lon_rx*_deg"]]

    output.interpolate("UnixTimeMillis",["LatitudeDegrees",
                                         "LongitudeDegrees"],inplace=True)
    return output

def solve_kaggle_dataset(folder_path, solver, verbose=False, *args, **kwargs):
    """Run solver on all kaggle traces.

    Additional ``*args`` arguments are passed into the ``solver``
    function.

    Parameters
    ----------
    folder_path: string or path-like
        Path to folder containing all traces (e.g. full path to "train"
        or "test" directories.
    solver : function
        State estimate solver that takes an instance of
        AndroidDerived2022 and outputs a state_estimate NavData object.
        Additional ``*args`` arguments are passed into this ``solver``
        function.
    verbose : bool
        If verbose, will print each trace trajectory name and phone name
	pair when it is solving the state estimate for that pair.

    Returns
    -------
    submission : gnss_lib_py.parsers.navdata.NavData
        Full solution submission across all traces. Can then be saved
        using submission.to_csv().

    """

    # create solution NavData object
    solution = NavData()

    # iterate through all trace options
    for trace_name in sorted(os.listdir(folder_path)):
        trace_path = os.path.join(folder_path, trace_name)
        # iterate through all phone types
        for phone_type in sorted(os.listdir(trace_path)):
            data_path = os.path.join(folder_path,trace_name,
                                     phone_type,"device_gnss.csv")
            try:
                # convert data to Measurement class
                derived_data = AndroidDerived2022(data_path)

                if verbose:
                    print("solving:",trace_name,phone_type)

                # compute state estimate using provided solver function
                state_estimate = solver(derived_data, *args, **kwargs)

                trip_id = "/".join([trace_name,phone_type])
                output = prepare_kaggle_submission(state_estimate,
                                                   trip_id)

                # concatenate solution to previous solutions
                solution.concat(navdata=output, inplace=True)

            except FileNotFoundError:
                continue

    return solution
