"""Functions to process Android measurements.

"""

__authors__ = "Ashwin Kanhere, Derek Knowles, Shubh Gupta, Adam Dai"
__date__ = "02 Nov 2021"


import os
import csv

import numpy as np
import pandas as pd

from gnss_lib_py.parsers.navdata import NavData
import gnss_lib_py.utils.constants as consts
from gnss_lib_py.utils.coordinates import wrap_0_to_2pi
from gnss_lib_py.utils.coordinates import geodetic_to_ecef
from gnss_lib_py.utils.coordinates import ecef_to_geodetic
from gnss_lib_py.utils.time_conversions import unix_to_gps_millis
from gnss_lib_py.utils.time_conversions import gps_to_unix_millis, get_leap_seconds

class AndroidRawGnss(NavData):
    """Handles Raw GNSS measurements from Android.

    Data types in the Android's GNSSStatus messages are documented on
    their website [1]_.

    References
    ----------
    .. [1] https://developer.android.com/reference/android/location/GnssStatus


    """
    def __init__(self, input_path, verbose=False, filter=False):
        """Android GNSSStatus file parser.

        Parameters
        ----------
        input_path : string or path-like
            Path to measurement csv file.
        filter : bool
            Filter noisy measurements based on known conditions.
        verbose : bool
            If true, prints extra debugging statements.

        """
        self.verbose = verbose
        self.filter = filter
        pd_df = self.preprocess(input_path)
        super().__init__(pandas_df=pd_df)


    def preprocess(self, input_path):
        """Built on the first parts of make_gnss_dataframe and correct_log

        """

        with open(input_path) as csvfile:
            reader = csv.reader(csvfile)
            row_idx = 0
            skip_rows = []
            header_row = None
            for row in reader:
                if len(row) == 0:
                    skip_rows.append(row_idx)
                    continue
                if row[0][0] == '#':
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

        print(measurements)
        for col in measurements.columns:
            print(col,measurements[col].dtype)

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
        qzss_idxs = self.argwhere("gnss_id","qzss")
        self["sv_id",qzss_idxs] = [consts.QZSS_PRN_SVN[i] for i in self.where("gnss_id","qzss")["sv_id"]]

        # add gps milliseconds
        self["gps_millis"] = unix_to_gps_millis(self["unix_millis"])

        # calculate pseudorange
        # Based off of MATLAB code from Google's gps-measurement-tools
        # repository: https://github.com/google/gps-measurement-tools. Compare
        # with opensource/ProcessGnssMeas.m
        gps_week_nanos = np.floor(-self["FullBiasNanos"]*1e-9/consts.WEEKSEC)*consts.WEEKSEC*1E9
        tx_rx_gnss_ns = self["TimeNanos"] - self["FullBiasNanos",0] - self["TimeOffsetNanos"] - self["BiasNanos"]

        t_rx_secs = np.zeros(len(self))

        # gps constellation
        tx_rx_gps_secs = (tx_rx_gnss_ns - gps_week_nanos)*1E-9
        t_rx_secs = np.where(self["gnss_id"]=="gps",
                             tx_rx_gps_secs,
                             t_rx_secs)
        # tx_rx_nanos = self["TimeNanos"] - self["FullBiasNanos",0] - gps_week_nanos
        # tx_rx_secs_gps = (tx_rx_nanos - self["TimeOffsetNanos"] - self["BiasNanos"])*1E-9,

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

        # add pseudorange uncertainty
        self["raw_pr_sigma_m"] = consts.C * 1E-9 * self["ReceivedSvTimeUncertaintyNanos"]

        if self.filter:
            self.filter_measurements(t_rx_secs)

    def filter_measurements(self,t_rx_secs):
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
        full_bias_filter = set(self.argwhere("FullBiasNanos",0,"geq"))
        filter_idxs = full_bias_filter


        # BiasUncertaintyNanos is too large
        bias_uncertainty_filter = set(self.argwhere("BiasUncertaintyNanos",40.,"geq"))
        print("bias_uncertainty_filter:",len(bias_uncertainty_filter))
        filter_idxs.update(bias_uncertainty_filter)

        # arrival time is negative or unrealistically large
        arrival_time_filter = set(np.argwhere(t_rx_secs>=1e7)[:,0])
        arrival_time_filter.update(set(np.argwhere(t_rx_secs<0)[:,0]))
        print("arrival_time_filter:",len(arrival_time_filter))
        filter_idxs.update(arrival_time_filter)

        # unknown constellations
        constellation_filter = set(self.argwhere("gnss_id","unknown"))
        print("constellation_filter:",len(constellation_filter))
        filter_idxs.update(constellation_filter)

        # TimeNanos is empty
        time_nanos_filter = set(self.argwhere("TimeNanos",np.nan))
        print("time_nanos_filter:",len(time_nanos_filter))
        filter_idxs.update(time_nanos_filter)

        # state is not STATE_TOW_DECODED.
        state_tow_decoded = 0x8
        state_filter = set(np.argwhere(~np.logical_and(self["State"],
                                                          state_tow_decoded))[:,0])
        print("state_filter:",len(state_filter))
        filter_idxs.update(state_filter)

        # ReceivedSvTimeUncertaintyNanos is too large
        received_uncertainty_filter = set(self.argwhere("ReceivedSvTimeUncertaintyNanos",500,"geq"))
        print("received_uncertainty_filter:",len(received_uncertainty_filter))
        filter_idxs.update(received_uncertainty_filter)

        # AdrState violates condition
        # ADR_STATE_VALID == 1 & ADR_STATE_RESET == 0
        # & ADR_STATE_CYCLE_SLIP == 0
        ADR_STATE_VALID = 0x1
        adr_valid = np.logical_and(self["AccumulatedDeltaRangeState"],ADR_STATE_VALID)
        ADR_STATE_RESET = 0x2
        adr_reset = ~(self["AccumulatedDeltaRangeState"] & ADR_STATE_RESET).astype(bool)
        ADR_STATE_CYCLE_SLIP = 0x4
        adr_slip = ~(self["AccumulatedDeltaRangeState"] & ADR_STATE_CYCLE_SLIP).astype(bool)
        adr_state_filter = set(np.argwhere(~np.logical_and(np.logical_and(adr_valid,adr_reset),adr_slip))[:,0])
        print("adr_state_filter:",len(adr_state_filter))
        filter_idxs.update(adr_state_filter)

        # AdrUncertaintyMeters is too large
        adr_uncertainty_filter = set(self.argwhere("AccumulatedDeltaRangeUncertaintyMeters",0.15,"geq"))
        print("adr_uncertainty_filter:",len(adr_uncertainty_filter))
        filter_idxs.update(adr_uncertainty_filter)

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
                   # "TimeNanos" : "",
                   # "LeapSecond" : "",
                   # "TimeUncertaintyNanos" : "",
                   # "FullBiasNanos" : "",
                   # "BiasNanos" : "",
                   # "BiasUncertaintyNanos" : "",
                   # "DriftNanosPerSecond" : "",
                   # "DriftUncertaintyNanosPerSecond" : "",
                   # "HardwareClockDiscontinuityCount" : "",
                   "Svid" : "sv_id",
                   # "TimeOffsetNanos" : "",
                   # "State" : "",
                   # "ReceivedSvTimeNanos" : "",
                   # "ReceivedSvTimeUncertaintyNanos" : "",
                   "Cn0DbHz" : "cn0_dbhz",
                   # "PseudorangeRateMetersPerSecond" : "",
                   # "PseudorangeRateUncertaintyMetersPerSecond" : "",
                   # "AccumulatedDeltaRangeState" : "",
                   # "AccumulatedDeltaRangeMeters" : "",
                   # "AccumulatedDeltaRangeUncertaintyMeters" : "",
                   # "CarrierFrequencyHz" : "",
                   # "CarrierCycles" : "",
                   # "CarrierPhase" : "",
                   # "CarrierPhaseUncertainty" : "",
                   # "MultipathIndicator" : "",
                   # "SnrInDb" : "",
                   "ConstellationType" : "gnss_id",
                   # "AgcDb" : "",
                   # "BasebandCn0DbHz" : "",
                   # "FullInterSignalBiasNanos" : "",
                   # "FullInterSignalBiasUncertaintyNanos" : "",
                   # "SatelliteInterSignalBiasNanos" : "",
                   # "SatelliteInterSignalBiasUncertaintyNanos" : "",
                   # "CodeType" : "signal_type",
                   # "ChipsetElapsedRealtimeNanos" : "",
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
        input_path : string or path-like
            File location of data file to read.

        Returns
        -------
        measurements : pd.DataFrame
            Dataframe that contains the accel and gyro measurements from
            the log.

        """

        if not isinstance(input_path, (str, os.PathLike)):
            raise TypeError("input_path must be string or path-like")
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path,"file not found")

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
            for row in reader:
                if row[0][0] == '#':
                    if 'Fix' in row[0]:
                        android_fixes = [row[1:]]
                else:
                    if row[0] == 'Fix':
                        android_fixes.append(row[1:])

        fix_df = pd.DataFrame(android_fixes[1:], columns = android_fixes[0])
        return fix_df
