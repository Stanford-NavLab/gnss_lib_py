"""Functions to process Android measurements.

Tested on Google Android's GNSSLogger App v3.0.6.4

"""

__authors__ = "Ashwin Kanhere, Derek Knowles, Shubh Gupta, Adam Dai"
__date__ = "02 Nov 2021"


import os
import csv

import numpy as np
import pandas as pd

from gnss_lib_py.parsers.navdata import NavData
from gnss_lib_py.algorithms.snapshot import solve_wls
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.sv_models import add_sv_states
from gnss_lib_py.utils.time_conversions import get_leap_seconds
from gnss_lib_py.utils.time_conversions import unix_to_gps_millis

class AndroidRawGnss(NavData):
    """Handles Raw GNSS measurements from Android.

    Data types in the Android's GNSSStatus messages are documented on
    their website [1]_.

    References
    ----------
    .. [1] https://developer.android.com/reference/android/location/GnssStatus


    """
    def __init__(self, input_path,
                 filter_measurements=True,
                 measurement_filters = {"bias_valid" : True,
                                        "bias_uncertainty" : 40.,
                                        "arrival_time" : True,
                                        "unknown_constellations" : True,
                                        "time_valid" : True,
                                        "state_decoded" : True,
                                        "sv_time_uncertainty" : 500.,
                                        "adr_valid" : True,
                                        "adr_uncertainty" : 15.
                                        },
                 verbose=False):
        """Android GNSSStatus file parser.

        Parameters
        ----------
        input_path : string or path-like
            Path to measurement csv file.
        filter_measurements : bool
            Filter noisy measurements based on known conditions.
        measurement_filters : dict
            Conditions under which measurements should be filtered. An
            emptry dictionary passed into measurement_filters is
            equivalent to setting filter_measurements to False.
        verbose : bool
            If true, prints extra debugging statements.

        """
        self.verbose = verbose
        self.filter_measurements = filter_measurements
        self.measurement_filters = measurement_filters
        pd_df = self.preprocess(input_path)
        super().__init__(pandas_df=pd_df)


    def preprocess(self, input_path):
        """Built on the first parts of make_gnss_dataframe and correct_log

        """

        with open(input_path, encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            row_idx = 0
            skip_rows = []
            header_row = None
            for row in reader:
                if len(row) == 0:
                    skip_rows.append(row_idx)
                elif len(row[0]) == 0:
                    skip_rows.append(row_idx)
                elif row[0][0] == '#':
                    if 'Raw' in row[0]:
                        header_row = row_idx
                    elif header_row is not None:
                        skip_rows.append(row_idx)
                elif row[0] != 'Raw':
                    skip_rows.append(row_idx)
                row_idx += 1

        measurements = pd.read_csv(input_path,
                                    skip_blank_lines = False,
                                    header = header_row,
                                    skiprows = skip_rows,
                                    dtype={'AccumulatedDeltaRangeUncertaintyMeters':np.float64},
                                    )

        return measurements

    def postprocess(self):
        """Postprocess loaded NavData.

        Arrival time taken from [5]_.

        References
        ----------
        .. [5] https://www.euspa.europa.eu/system/files/reports/gnss_raw_measurement_web_0.pdf

        """

        # rename gnss_id
        gnss_id = np.array([consts.CONSTELLATION_ANDROID[i] for i in self["gnss_id"]])
        self["gnss_id"] = gnss_id

        # update svn for QZSS constellation
        if "qzss" in np.unique(self["gnss_id"]):
            qzss_idxs = self.argwhere("gnss_id","qzss")
            self["sv_id",qzss_idxs] = [consts.QZSS_PRN_SVN[i] \
                        for i in self.where("gnss_id","qzss")["sv_id"]]

        # add singal type information where available
        self["signal_type"] = np.array([consts.CODE_TYPE_ANDROID[x].get(y,"") \
                                        for x,y in zip(self["gnss_id"],
                                                       self["CodeType"])])

        # add gps milliseconds
        self["gps_millis"] = unix_to_gps_millis(self["unix_millis"])

        # calculate pseudorange
        # Based off of MATLAB code from Google's gps-measurement-tools
        # repository: https://github.com/google/gps-measurement-tools. Compare
        # with opensource/ProcessGnssMeas.m
        gps_week_nanos = np.floor(-self["FullBiasNanos"]*1e-9/consts.WEEKSEC)*consts.WEEKSEC*1E9
        tx_rx_gnss_ns = self["TimeNanos"] - self["FullBiasNanos",0] + self["TimeOffsetNanos"] - self["BiasNanos",0]

        t_rx_secs = np.zeros(len(self))

        # gps constellation
        tx_rx_gps_secs = (tx_rx_gnss_ns - gps_week_nanos)*1E-9
        t_rx_secs = np.where(self["gnss_id"]=="gps",
                             tx_rx_gps_secs,
                             t_rx_secs)

        # beidou constellation
        tx_rx_beidou_secs = (tx_rx_gnss_ns - gps_week_nanos)*1E-9 - 14.
        t_rx_secs = np.where(self["gnss_id"]=="beidou",
                             tx_rx_beidou_secs,
                             t_rx_secs)

        # galileo constellation
        # nanos_per_100ms = 100*1E6
        # ms_number_nanos = np.floor(-self["FullBiasNanos"]/nanos_per_100ms)*nanos_per_100ms
        # tx_rx_galileo_secs = (tx_rx_gnss_ns - ms_number_nanos)*1E-9
        t_rx_secs = np.where(self["gnss_id"]=="galileo",
                             tx_rx_gps_secs,
                             t_rx_secs)

        # glonass constellation
        nanos_per_day = 1E9*24*60*60
        day_number_nanos = np.floor(-self["FullBiasNanos"]/nanos_per_day)*nanos_per_day
        tx_rx_glonass_secs = (tx_rx_gnss_ns - day_number_nanos)*1E-9\
                           + 3*60*60 - get_leap_seconds(self["gps_millis",0].item())
        t_rx_secs = np.where(self["gnss_id"]=="glonass",
                             tx_rx_glonass_secs,
                             t_rx_secs)

        # qzss constellation
        tx_rx_qzss_secs = (tx_rx_gnss_ns - gps_week_nanos)*1E-9
        t_rx_secs = np.where(self["gnss_id"]=="qzss",
                             tx_rx_qzss_secs,
                             t_rx_secs)

        t_tx_secs = self["ReceivedSvTimeNanos"]*1E-9
        self["raw_pr_m"] = (t_rx_secs - t_tx_secs)*consts.C

        # remove the receiver's clock bias at the first timestamp
        for _, _, subset in self.loop_time("gps_millis", delta_t_decimals=-2):
            subset = add_sv_states(subset.where("gnss_id",("gps","galileo")),source="precise")
            subset["corr_pr_m"] = subset["raw_pr_m"] + subset["b_sv_m"]
            first_timestamp = solve_wls(subset)
            if not np.isnan(first_timestamp["b_rx_wls_m"]):
                break
        self["raw_pr_m"] += first_timestamp["b_rx_wls_m"]

        # add pseudorange uncertainty
        self["raw_pr_sigma_m"] = consts.C * 1E-9 * self["ReceivedSvTimeUncertaintyNanos"]

        if self.filter_measurements:
            self.filter_raw_measurements(t_rx_secs)

    def filter_raw_measurements(self,t_rx_secs):
        """Filter noisy measurements.


        The State variable options are shown on [4]_.

        References
        ----------
        .. [1] Michael Fu, Mohammed Khider, Frank van Diggelen, Dave
               Orendorff. "Workshop for Google Smartphone Decimeter
               Challenge (SDC) 2023-2024." ION GNSS+ 2023.
        .. [2] https://github.com/google/gps-measurement-tools/blob/master/opensource/ProcessGnssMeas.m
        .. [3] https://github.com/google/gps-measurement-tools/blob/master/opensource/SetDataFilter.m
        .. [4] https://developer.android.com/reference/android/location/GnssMeasurement#STATE_TOW_DECODED
        """

        # FullBiasNanos is zero or invalid
        if "bias_valid" in self.measurement_filters \
            and self.measurement_filters["bias_valid"]:
            full_bias_filter = set(self.argwhere("FullBiasNanos",0,"geq"))
            filter_idxs = full_bias_filter
            if self.verbose:
                print("bias_valid removed",len(full_bias_filter))

        # BiasUncertaintyNanos is too large
        if "bias_uncertainty" in self.measurement_filters:
            bias_uncertainty_filter = set(self.argwhere("BiasUncertaintyNanos",
                                          self.measurement_filters["bias_uncertainty"],"geq"))
            filter_idxs.update(bias_uncertainty_filter)
            if self.verbose:
                print("bias_uncertainty_filter removed",
                      len(bias_uncertainty_filter))

        # arrival time is negative or unrealistically large
        if "arrival_time" in self.measurement_filters \
            and self.measurement_filters["arrival_time"]:
            arrival_time_filter = set(np.argwhere(t_rx_secs>=1e7)[:,0])
            arrival_time_filter.update(set(np.argwhere(t_rx_secs<0)[:,0]))
            filter_idxs.update(arrival_time_filter)
            if self.verbose:
                print("arrival_time_filter removed",
                      len(arrival_time_filter))

        # unknown constellations
        if "unknown_constellations" in self.measurement_filters \
            and self.measurement_filters["unknown_constellations"]:
            constellation_filter = set(self.argwhere("gnss_id","unknown"))
            filter_idxs.update(constellation_filter)
            if self.verbose:
                print("constellation_filter removed",
                      len(constellation_filter))

        # TimeNanos is empty
        if "time_valid" in self.measurement_filters \
            and self.measurement_filters["time_valid"]:
            time_nanos_filter = set(self.argwhere("TimeNanos",np.nan))
            filter_idxs.update(time_nanos_filter)
            if self.verbose:
                print("time_nanos_filter removed",
                      len(time_nanos_filter))

        # state is not STATE_TOW_DECODED.
        if "state_decoded" in self.measurement_filters \
            and self.measurement_filters["state_decoded"]:
            state_tow_decoded = 0x8
            state_filter = set(np.argwhere(~np.logical_and(self["State"],
                                                              state_tow_decoded))[:,0])
            filter_idxs.update(state_filter)
            if self.verbose:
                print("state_filter removed",len(state_filter))

        # ReceivedSvTimeUncertaintyNanos is too large
        if "sv_time_uncertainty" in self.measurement_filters:
            received_uncertainty_filter = set(self.argwhere("ReceivedSvTimeUncertaintyNanos",
                                            self.measurement_filters["sv_time_uncertainty"],"geq"))
            filter_idxs.update(received_uncertainty_filter)
            if self.verbose:
                print("received_uncertainty_filter removed",
                      len(received_uncertainty_filter))

        # AdrState violates condition
        # ADR_STATE_VALID == 1 & ADR_STATE_RESET == 0
        # & ADR_STATE_CYCLE_SLIP == 0
        if "adr_valid" in self.measurement_filters \
            and self.measurement_filters["adr_valid"]:
            ADR_STATE_VALID = 0x1
            adr_valid = np.logical_and(self["AccumulatedDeltaRangeState"],ADR_STATE_VALID)
            ADR_STATE_RESET = 0x2
            adr_reset = ~(self["AccumulatedDeltaRangeState"] & ADR_STATE_RESET).astype(bool)
            ADR_STATE_CYCLE_SLIP = 0x4
            adr_slip = ~(self["AccumulatedDeltaRangeState"] & ADR_STATE_CYCLE_SLIP).astype(bool)
            adr_state_filter = set(np.argwhere(~np.logical_and(np.logical_and(adr_valid,
                                    adr_reset),adr_slip))[:,0])
            filter_idxs.update(adr_state_filter)
            if self.verbose:
                print("adr_state_filter removed",len(adr_state_filter))

        # adr_uncertainty is too large
        if "adr_uncertainty" in self.measurement_filters:
            adr_uncertainty_filter = set(self.argwhere("accumulated_delta_range_sigma_m",
                                    self.measurement_filters["adr_uncertainty"],"geq"))
            filter_idxs.update(adr_uncertainty_filter)
            if self.verbose:
                print("adr_uncertainty_filter removed",
                      len(adr_uncertainty_filter))

        # removed filtered measurements
        filter_idxs = sorted(list(filter_idxs))
        self.remove(cols=filter_idxs,inplace=True)

    @staticmethod
    def _row_map():
        """Map of column names from loaded to gnss_lib_py standard

        Returns
        -------
        row_map : Dict
            Dictionary of the form {old_name : new_name}

        """

        row_map = {
                   "utcTimeMillis" : "unix_millis",
                   "Svid" : "sv_id",
                   "Cn0DbHz" : "cn0_dbhz",
                   "AccumulatedDeltaRangeMeters" : "accumulated_delta_range_m",
                   "AccumulatedDeltaRangeUncertaintyMeters" : "accumulated_delta_range_sigma_m",
                   "ConstellationType" : "gnss_id",
                  }

        return row_map

class AndroidRawAccel(NavData):
    """Class handling Accelerometer measurements from Android.

    Inherits from NavData().
    """
    def __init__(self, input_path,
                 sensor_fields=("UncalAccel","Accel")):

        self.sensor_fields = sensor_fields
        pd_df = self.preprocess(input_path)
        super().__init__(pandas_df=pd_df)

    def preprocess(self, input_path):
        """Read Android raw file and produce Accel dataframe objects.

        Parameters
        ----------
        input_path : string or path-like
            File location of data file to read.

        Returns
        -------
        measurements : pd.DataFrame
            Dataframe that contains the accel measurements from the log.

        """

        if not isinstance(input_path, (str, os.PathLike)):
            raise TypeError("input_path must be string or path-like")
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path,"file not found")

        sensor_data = {}

        with open(input_path, encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) == 0 or len(row[0]) == 0:
                    continue
                if row[0][0] == '#':    # header row
                    if len(row) == 1:
                        continue
                    sensor_field = row[0][2:]
                    if sensor_field in self.sensor_fields:
                        sensor_data[sensor_field] = [row[1:]]
                else:
                    if row[0] in self.sensor_fields:
                        sensor_data[row[0]].append(row[1:])

        sensor_dfs = [pd.DataFrame(data[1:], columns = data[0],
                                   dtype=np.float64) for _,data in sensor_data.items()]

        # remove empty dataframes
        sensor_dfs = [df for df in sensor_dfs if len(df) > 0]

        if len(sensor_dfs) == 0:
            measurements = pd.DataFrame()
        elif len(sensor_dfs) > 1:
            measurements = pd.concat(sensor_dfs, axis=1)
        else:
            measurements = sensor_dfs[0]

        return measurements

    def postprocess(self):
        """Postprocess loaded data."""

        # add gps milliseconds
        self["gps_millis"] = unix_to_gps_millis(self["unix_millis"])

    def _row_map(self):
        row_map = {
                   'utcTimeMillis' : 'unix_millis',
                   'AccelXMps2' : 'acc_x_mps2',
                   'AccelYMps2' : 'acc_y_mps2',
                   'AccelZMps2' : 'acc_z_mps2',
                   'UncalAccelXMps2' : 'acc_x_uncal_mps2',
                   'UncalAccelYMps2' : 'acc_y_uncal_mps2',
                   'UncalAccelZMps2' : 'acc_z_uncal_mps2',
                   'BiasXMps2' : 'acc_bias_x_mps2',
                   'BiasYMps2' : 'acc_bias_y_mps2',
                   'BiasZMps2' : 'acc_bias_z_mps2',
                   }
        row_map = {k:v for k,v in row_map.items() if k in self.rows}
        return row_map

class AndroidRawGyro(AndroidRawAccel):
    """Class handling Gyro measurements from Android.

    """
    def __init__(self, input_path):
        sensor_fields = ("UncalGyro","Gyro")
        super().__init__(input_path, sensor_fields=sensor_fields)

    def _row_map(self):
        row_map = {
                   'utcTimeMillis' : 'unix_millis',
                   'GyroXRadPerSec' : 'ang_vel_x_radps',
                   'GyroYRadPerSec' : 'ang_vel_y_radps',
                   'GyroZRadPerSec' : 'ang_vel_z_radps',
                   'UncalGyroXRadPerSec' : 'ang_vel_x_uncal_radps',
                   'UncalGyroYRadPerSec' : 'ang_vel_y_uncal_radps',
                   'UncalGyroZRadPerSec' : 'ang_vel_z_uncal_radps',
                   'DriftXMps2' : 'ang_vel_drift_x_radps',
                   'DriftYMps2' : 'ang_vel_drift_y_radps',
                   'DriftZMps2' : 'ang_vel_drift_z_radps',
                   }
        row_map = {k:v for k,v in row_map.items() if k in self.rows}
        return row_map

class AndroidRawMag(AndroidRawAccel):
    """Class handling Magnetometer measurements from Android.

    """
    def __init__(self, input_path):
        sensor_fields = ("UncalMag","Mag")
        super().__init__(input_path, sensor_fields=sensor_fields)

    def _row_map(self):
        row_map = {
                   'utcTimeMillis' : 'unix_millis',
                   'MagXMicroT' : 'mag_x_microt',
                   'MagYMicroT' : 'mag_y_microt',
                   'MagZMicroT' : 'mag_z_microt',
                   'UncalMagXMicroT' : 'mag_x_uncal_microt',
                   'UncalMagYMicroT' : 'mag_y_uncal_microt',
                   'UncalMagZMicroT' : 'mag_z_uncal_microt',
                   'BiasXMicroT' : 'mag_bias_x_microt',
                   'BiasYMicroT' : 'mag_bias_y_microt',
                   'BiasZMicroT' : 'mag_bias_z_microt',
                   }
        row_map = {k:v for k,v in row_map.items() if k in self.rows}
        return row_map

class AndroidRawOrientation(AndroidRawAccel):
    """Class handling Orientation measurements from Android.

    """
    def __init__(self, input_path):
        sensor_fields = ("OrientationDeg")
        super().__init__(input_path, sensor_fields=sensor_fields)

    def _row_map(self):
        row_map = {
                   'utcTimeMillis' : 'unix_millis',
                   'yawDeg' : 'yaw_rx_deg',
                   'rollDeg' : 'roll_rx_deg',
                   'pitchDeg' : 'pitch_rx_deg',
                   }
        row_map = {k:v for k,v in row_map.items() if k in self.rows}
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
        input_path : string or path-like
            File location of data file to read.

        Returns
        -------
        fix_df : pd.DataFrame
            Dataframe that contains the location fixes from the log.

        """

        if not isinstance(input_path, (str, os.PathLike)):
            raise TypeError("input_path must be string or path-like")
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path,"file not found")

        with open(input_path, encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            row_idx = 0
            skip_rows = []
            header_row = None
            for row in reader:
                if len(row) == 0:
                    skip_rows.append(row_idx)
                elif len(row[0]) == 0:
                    skip_rows.append(row_idx)
                elif row[0][0] == '#':
                    if 'Fix' in row[0]:
                        header_row = row_idx
                    elif header_row is not None:
                        skip_rows.append(row_idx)
                elif row[0] != 'Fix':
                    skip_rows.append(row_idx)
                row_idx += 1

        fix_df = pd.read_csv(input_path,
                             skip_blank_lines = False,
                             header = header_row,
                             skiprows = skip_rows,
                             )

        return fix_df


    def postprocess(self):
        """Postprocess loaded data.

        """

        # add gps milliseconds
        self["gps_millis"] = unix_to_gps_millis(self["unix_millis"])

        # rename provider
        self["fix_provider"] = np.array([self._provider_map().get(i,"")\
                                         for i in self["fix_provider"]])

        # add heading in radians
        self["heading_rx_rad"] = np.deg2rad(self["heading_rx_deg"])

    @staticmethod
    def _row_map():
        row_map = {"LatitudeDegrees" : "lat_rx_deg",
                   "LongitudeDegrees" : "lon_rx_deg",
                   "AltitudeMeters" : "alt_rx_m",
                   "Provider" : "fix_provider",
                   "BearingDegrees" : "heading_rx_deg",
                   "UnixTimeMillis" : "unix_millis",
                   }
        return row_map

    @staticmethod
    def _provider_map():
        provider_map = {"FLP" : "fused",
                        "GPS" : "gnss",
                        "NLP" : "network",
                        }
        return provider_map
